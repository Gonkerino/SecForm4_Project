-- Run once with psql as the PostgreSQL administrator. Passwords are prompted.
-- Existing roles/databases cause an error; nothing is reset or overwritten.
\set ON_ERROR_STOP on
CREATE ROLE insider LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE;
\password insider
CREATE DATABASE insider_research OWNER insider;
