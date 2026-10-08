CREATE DATABASE IF NOT EXISTS recipeai;
CREATE USER IF NOT EXISTS 'hari'@'localhost' IDENTIFIED BY '1234';
GRANT ALL PRIVILEGES ON recipeai.* TO 'hari'@'localhost';

-- Apply the changes
FLUSH PRIVILEGES;

-- Use the database
USE recipeai;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
-- Create index on email for faster lookups
CREATE INDEX IF NOT EXISTS idx_email ON users(email);
SELECT 'Database and user setup completed successfully!' AS Status;
