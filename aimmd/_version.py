# This file is part of aimmd
#
# aimmd is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# aimmd is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with aimmd. If not, see <https://www.gnu.org/licenses/>.
"""
Helper functions to populate aimmd.__version__ and aimmd.__git_hash__.

If we are in a git-repository (and detect commits since the last version-tagged
commit) we will add a git-hash to the version.
"""
import os
import subprocess
import importlib.metadata


def _get_git_hash_and_tag():
    """Get git hash, date, and tag from git log."""
    git_hash = ""
    git_date = ""
    git_tag = ""
    with subprocess.Popen(
            ["git", "log", "-1", "--format='%H || %as || %(describe:tags=true,match=v*)'"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(__file__),
                          ) as p:
        stdout, _ = p.communicate()
        returncode = p.returncode
    if not returncode:
        git_hash, git_date, git_describe = (stdout.decode("utf-8")
                                            .replace("'", "").replace('"', '')
                                            .strip().split("||"))
        git_date = git_date.strip().replace("-", "")
        git_describe = git_describe.strip()
        if git_describe and "-" not in git_describe:
            # git-describe returns either the git-tag or (if we are not exactly
            #  at a tag) something like
            #  $GITTAG-$NUM_COMMITS_DISTANCE-$CURRENT_COMMIT_HASH
            git_tag = git_describe[1:]  # strip of the 'v'
    return git_hash, git_date, git_tag


_version = importlib.metadata.version("aimmd")
_git_hash, _git_date, _git_tag = _get_git_hash_and_tag()
__git_hash__ = _git_hash
if _version == _git_tag or not _git_hash:
    # dont append git_hash to version, if it is a version-tagged commit or if
    # git_hash is empty (happens if git is installed but we are not in a repo)
    __version__ = _version
else:
    __version__ = _version + f"+git{_git_date}.{_git_hash[:7]}"
