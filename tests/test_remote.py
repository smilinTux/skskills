"""Tests for SKSkills Remote Registry — package, download, publish, git clone."""

import json
import tarfile
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch

import pytest

from skskills.remote import RemoteRegistry, RemoteSkillEntry, RegistryIndex


@pytest.fixture
def skill_source(tmp_path: Path) -> Path:
    """Create a minimal skill directory for testing."""
    skill_dir = tmp_path / "test-skill-src"
    skill_dir.mkdir()
    (skill_dir / "knowledge").mkdir()
    (skill_dir / "knowledge" / "SKILL.md").write_text("# Test\n")
    (skill_dir / "tools").mkdir()

    yaml_content = dedent("""\
        name: test-skill
        version: "0.1.0"
        description: A test skill for remote registry
        author:
          name: tester
        knowledge:
          - path: knowledge/SKILL.md
            description: Test context
            auto_load: true
        tags:
          - test
          - remote
    """)
    (skill_dir / "skill.yaml").write_text(yaml_content)
    return skill_dir


@pytest.fixture
def remote(tmp_path: Path) -> RemoteRegistry:
    """Create a remote registry with a temp cache."""
    return RemoteRegistry(
        registry_url="https://test-registry.example.com/api",
        cache_dir=tmp_path / "cache",
    )


class TestPackage:
    """Test skill packaging into tarballs."""

    def test_package_creates_tarball(self, skill_source: Path, tmp_path: Path):
        """Packaging a skill should create a .tar.gz file."""
        tarball = RemoteRegistry.package(skill_source, tmp_path / "output")
        assert tarball.exists()
        assert tarball.name == "test-skill-0.1.0.tar.gz"

    def test_package_creates_metadata_sidecar(self, skill_source: Path, tmp_path: Path):
        """Packaging should create a .json metadata file alongside the tarball."""
        tarball = RemoteRegistry.package(skill_source, tmp_path / "output")
        meta_path = tarball.with_suffix(".json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["name"] == "test-skill"
        assert meta["version"] == "0.1.0"
        assert "sha256" in meta
        assert len(meta["sha256"]) == 64  # SHA-256 hex digest

    def test_package_contains_skill_yaml(self, skill_source: Path, tmp_path: Path):
        """The tarball should contain skill.yaml."""
        tarball = RemoteRegistry.package(skill_source, tmp_path / "output")
        with tarfile.open(tarball, "r:gz") as tar:
            names = tar.getnames()
            assert "skill.yaml" in names

    def test_package_excludes_hidden_files(self, skill_source: Path, tmp_path: Path):
        """Hidden files should be excluded from the package."""
        (skill_source / ".git").mkdir()
        (skill_source / ".git" / "config").write_text("git config")
        tarball = RemoteRegistry.package(skill_source, tmp_path / "output")
        with tarfile.open(tarball, "r:gz") as tar:
            names = tar.getnames()
            assert not any(".git" in n for n in names)

    def test_package_missing_skill_yaml(self, tmp_path: Path):
        """Packaging a directory without skill.yaml should fail."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            RemoteRegistry.package(empty_dir)


class TestRegistryIndex:
    """Test remote index fetching and caching."""

    def test_fetch_index_with_cache_fallback(self, remote: RemoteRegistry):
        """When remote is unreachable but cache exists, use cache."""
        # Seed the cache
        index = RegistryIndex(
            url=remote.registry_url,
            skills=[
                RemoteSkillEntry(name="cached-skill", version="1.0.0"),
            ],
            fetched_at="2026-01-01T00:00:00",
        )
        cache_file = remote.cache_dir / "index.json"
        cache_file.write_text(index.model_dump_json(indent=2))

        # Mock HTTP to fail
        with patch.object(remote, "_http_get", side_effect=ConnectionError("offline")):
            result = remote.fetch_index()
            assert len(result.skills) == 1
            assert result.skills[0].name == "cached-skill"

    def test_fetch_index_no_cache_raises(self, remote: RemoteRegistry):
        """When remote is unreachable and no cache, should raise."""
        with patch.object(remote, "_http_get", side_effect=ConnectionError("offline")):
            with pytest.raises(ConnectionError):
                remote.fetch_index()


class TestSearch:
    """Test remote search functionality."""

    def test_search_by_name(self, remote: RemoteRegistry):
        """Search should match on skill name."""
        remote._index_cache = RegistryIndex(
            url=remote.registry_url,
            skills=[
                RemoteSkillEntry(name="syncthing-setup", version="1.0.0", description="Syncthing"),
                RemoteSkillEntry(name="capauth-login", version="1.0.0", description="CapAuth"),
            ],
        )
        results = remote.search("syncthing")
        assert len(results) == 1
        assert results[0].name == "syncthing-setup"

    def test_search_by_tag(self, remote: RemoteRegistry):
        """Search should match on tags."""
        remote._index_cache = RegistryIndex(
            url=remote.registry_url,
            skills=[
                RemoteSkillEntry(name="my-skill", version="1.0.0", tags=["sovereign", "auth"]),
            ],
        )
        results = remote.search("sovereign")
        assert len(results) == 1

    def test_search_no_results(self, remote: RemoteRegistry):
        """Search with no matches should return empty list."""
        remote._index_cache = RegistryIndex(url=remote.registry_url, skills=[])
        assert remote.search("nonexistent") == []


class TestGetSkillInfo:
    """Test skill info lookup."""

    def test_get_latest_version(self, remote: RemoteRegistry):
        """Should return the latest version when version is None."""
        remote._index_cache = RegistryIndex(
            url=remote.registry_url,
            skills=[
                RemoteSkillEntry(name="my-skill", version="0.1.0"),
                RemoteSkillEntry(name="my-skill", version="1.0.0"),
                RemoteSkillEntry(name="my-skill", version="0.5.0"),
            ],
        )
        entry = remote.get_skill_info("my-skill")
        assert entry is not None
        assert entry.version == "1.0.0"

    def test_get_specific_version(self, remote: RemoteRegistry):
        """Should return the requested version."""
        remote._index_cache = RegistryIndex(
            url=remote.registry_url,
            skills=[
                RemoteSkillEntry(name="my-skill", version="0.1.0"),
                RemoteSkillEntry(name="my-skill", version="1.0.0"),
            ],
        )
        entry = remote.get_skill_info("my-skill", version="0.1.0")
        assert entry is not None
        assert entry.version == "0.1.0"

    def test_get_nonexistent(self, remote: RemoteRegistry):
        """Missing skill should return None."""
        remote._index_cache = RegistryIndex(url=remote.registry_url, skills=[])
        assert remote.get_skill_info("nonexistent") is None


class TestDownload:
    """Test skill download and extraction."""

    def test_download_and_extract(self, remote: RemoteRegistry, skill_source: Path, tmp_path: Path):
        """Downloading should extract to cache directory."""
        # Create a tarball to serve
        tarball = RemoteRegistry.package(skill_source, tmp_path / "serve")
        sha256 = __import__("hashlib").sha256(tarball.read_bytes()).hexdigest()

        remote._index_cache = RegistryIndex(
            url=remote.registry_url,
            skills=[
                RemoteSkillEntry(
                    name="test-skill",
                    version="0.1.0",
                    download_url=f"file://{tarball}",
                    sha256=sha256,
                ),
            ],
        )

        # Mock HTTP download to copy the file
        def mock_download(url, dest):
            import shutil
            shutil.copy2(tarball, dest)
            return dest

        with patch.object(remote, "_http_download", side_effect=mock_download):
            result = remote.download("test-skill")
            assert (result / "skill.yaml").exists()

    def test_download_checksum_mismatch(self, remote: RemoteRegistry, skill_source: Path, tmp_path: Path):
        """Bad checksum should raise ValueError."""
        tarball = RemoteRegistry.package(skill_source, tmp_path / "serve")

        remote._index_cache = RegistryIndex(
            url=remote.registry_url,
            skills=[
                RemoteSkillEntry(
                    name="test-skill",
                    version="0.1.0",
                    download_url=f"file://{tarball}",
                    sha256="badhash",
                ),
            ],
        )

        def mock_download(url, dest):
            import shutil
            shutil.copy2(tarball, dest)
            return dest

        with patch.object(remote, "_http_download", side_effect=mock_download):
            with pytest.raises(ValueError, match="Checksum mismatch"):
                remote.download("test-skill")

    def test_download_not_found(self, remote: RemoteRegistry):
        """Downloading a nonexistent skill should raise FileNotFoundError."""
        remote._index_cache = RegistryIndex(url=remote.registry_url, skills=[])
        with pytest.raises(FileNotFoundError, match="not found"):
            remote.download("nonexistent")


class TestFromGit:
    """Test git clone installation."""

    def test_from_git_success(self, skill_source: Path, tmp_path: Path):
        """Successful git clone should return the skill directory."""
        dest = tmp_path / "cloned"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Pre-populate the dest with skill files (simulating git clone)
            import shutil
            shutil.copytree(skill_source, dest)

            result = RemoteRegistry.from_git("https://github.com/test/skill.git", dest)
            assert (result / "skill.yaml").exists()

    def test_from_git_failure(self, tmp_path: Path):
        """Failed git clone should raise RuntimeError."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="fatal: repo not found")
            with pytest.raises(RuntimeError, match="git clone failed"):
                RemoteRegistry.from_git("https://github.com/test/nonexistent.git", tmp_path / "dest")
