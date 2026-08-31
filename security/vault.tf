# vault-config.hcl

# Enable AppRole Auth Method
resource "vault_auth_backend" "approle" {
  type        = "approle"
  description = "AppRole authentication for API"
}

# Create a Vault policy for the API
resource "vault_policy" "api_policy" {
  name = "api-policy"
  policy = <<EOT
# Allow API to read its secrets
path "secret/data/api/*" {
  capabilities = ["read"]
}
EOT
}

# Configure the AppRole role for the API
resource "vault_approle_auth_backend_role" "api_role" {
  backend        = vault_auth_backend.approle.path
  role_name      = "api-role"
  token_policies = ["default", vault_policy.api_policy.name]
}

# Enable KV v2 Secrets Engine
resource "vault_mount" "kv" {
  path        = "secret"
  type        = "kv"
  options     = { version = "2" }
  description = "KV Version 2 secret engine mount for API"
}

# Add an example secret for the API
resource "vault_kv_secret_v2" "api_secret" {
  mount       = vault_mount.kv.path
  name        = "api/config"
  cas         = 1
  delete_all_versions = true
  data_json = jsonencode(
    {
      "db_password" = "super-secret-password",
      "api_key"     = "api-secret-key"
    }
  )
}
