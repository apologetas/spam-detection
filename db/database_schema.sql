
CREATE TABLE IF NOT EXISTS spam_model (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_weights BYTEA NOT NULL,
    vectorizer BYTEA NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE INDEX IF NOT EXISTS idx_spam_model_name_date ON spam_model(model_name, created_at);