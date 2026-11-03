-- Database-level storage contract; existing catalogues remain full until migrated.
INSERT INTO metadata(key,value) VALUES ('storage_profile','full');
UPDATE metadata SET value='5' WHERE key='schema_version';
