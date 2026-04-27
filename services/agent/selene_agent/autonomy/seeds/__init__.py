"""Idempotent seeders for system-owned agenda items.

Each module here exposes ``ensure_seeds()`` — call it once at startup. Seed
rows are marked ``created_by='system_camera'`` (or similar) so manual edits
to user-owned rows are never overwritten.
"""
