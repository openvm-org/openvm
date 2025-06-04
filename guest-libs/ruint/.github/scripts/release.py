#!/usr/bin/env python3

# Steps:
# * Update `Cargo.lock`
# * Retrieve current version from `Cargo.toml`
# * Retrieve unreleased changes from the `Changelog.md`
# * Stop if no changes
# * Determine if it is a major, minor or patch release
# * Create new version number
# * Update all `Cargo.toml`s in workspace
# * Update `Changelog.md`
#   * Rename existing 'Unreleased' section and create new.
#   * Update links at the end.
# * Git commit changes
# * Git tag with version
# * Git push branch and tag
# * Cargo publish in all workspace members
