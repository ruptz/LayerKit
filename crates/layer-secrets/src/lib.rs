//! Secrets for the Layer apps — app keys and sign-ins — kept in the system
//! keystore rather than in the app's JSON files: Windows Credential Manager,
//! the macOS Keychain, or the Secret Service (GNOME Keyring, KWallet) on Linux.
//!
//! The keystore encrypts them for the signed-in user, so a copy of the app's
//! folder — a backup, a synced AppData, a settings file pasted into a support
//! chat — carries no secrets.
//!
//! Some machines have no keystore (a Linux desktop without a keyring daemon,
//! a locked keychain). `Vault::available` says so, and the app keeps its
//! secrets in its own files there instead, as it did before: never broken for
//! want of a keystore.

/// One app's secrets, each under a name of its own.
#[derive(Clone, Debug)]
pub struct Vault {
    service: String,
    available: bool,
}

impl Vault {
    /// `service` is what the keystore lists them under, e.g. "JammyLayer".
    /// Checks once whether the keystore can be used.
    pub fn new(service: &str) -> Self {
        let available = keyring::Entry::new(service, "probe").is_ok_and(|e| match e.get_password() {
            Ok(_) | Err(keyring::Error::NoEntry) => true,
            Err(_) => false,
        });
        Self { service: service.to_owned(), available }
    }

    /// One that never touches the keystore: everything stays in the app's
    /// files. For machines that don't have one, and for tests.
    pub fn unavailable(service: &str) -> Self {
        Self { service: service.to_owned(), available: false }
    }

    pub fn available(&self) -> bool {
        self.available
    }

    /// None when there's no such secret, or no keystore.
    pub fn get(&self, name: &str) -> Option<String> {
        if !self.available {
            return None;
        }
        keyring::Entry::new(&self.service, name).ok()?.get_password().ok().filter(|v| !v.is_empty())
    }

    /// Stores `value`, or forgets the secret when it's empty. False when the
    /// keystore couldn't take it, so the caller keeps it somewhere else.
    pub fn set(&self, name: &str, value: &str) -> bool {
        if !self.available {
            return false;
        }
        let Ok(entry) = keyring::Entry::new(&self.service, name) else { return false };
        if value.is_empty() {
            matches!(entry.delete_credential(), Ok(()) | Err(keyring::Error::NoEntry))
        } else {
            entry.set_password(value).is_ok()
        }
    }

    pub fn delete(&self, name: &str) -> bool {
        self.set(name, "")
    }

    /// A secret the app is reading from its own file. Left there when there's
    /// no keystore. Otherwise a value still in the file is moved into the
    /// keystore, and true is returned so the app rewrites the file without it;
    /// an empty one is filled from the keystore.
    pub fn adopt(&self, name: &str, value: &mut String) -> bool {
        if !self.available {
            return false;
        }
        if value.is_empty() {
            *value = self.get(name).unwrap_or_default();
            false
        } else {
            self.set(name, value)
        }
    }

    /// A secret the app is about to write to its own file: stored in the
    /// keystore and blanked in the file's copy, or left there if the
    /// keystore couldn't take it.
    pub fn keep(&self, name: &str, value: &mut String) {
        if self.set(name, value) {
            value.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vault() -> Vault {
        Vault::new(&format!("layer-secrets-test-{}", std::process::id()))
    }

    #[test]
    fn without_a_keystore_everything_stays_in_the_file() {
        let vault = Vault::unavailable("anything");
        let mut value = "secret".to_owned();
        assert!(!vault.adopt("a", &mut value));
        vault.keep("a", &mut value);
        assert_eq!(value, "secret");
        assert_eq!(vault.get("a"), None);
    }

    /// Skipped where the machine has no keystore, as CI's Linux runner doesn't.
    #[test]
    fn moves_secrets_out_of_the_file_and_back() {
        let vault = vault();
        if !vault.available() {
            eprintln!("no keystore here; skipped");
            return;
        }

        // A secret still in the file moves into the keystore...
        let mut from_file = "s3cret".to_owned();
        assert!(vault.adopt("name", &mut from_file));
        assert_eq!(from_file, "s3cret");
        // ...and is read from there once the file no longer has it.
        let mut from_file = String::new();
        assert!(!vault.adopt("name", &mut from_file));
        assert_eq!(from_file, "s3cret");

        // Writing keeps it out of the file.
        let mut to_file = "changed".to_owned();
        vault.keep("name", &mut to_file);
        assert!(to_file.is_empty());
        assert_eq!(vault.get("name").as_deref(), Some("changed"));

        // Empty forgets it.
        assert!(vault.delete("name"));
        assert_eq!(vault.get("name"), None);
        assert!(vault.delete("name"), "forgetting twice is fine");
    }
}
