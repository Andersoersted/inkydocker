# InkyDocker Security Documentation

## Security Improvements Implemented

### 🛡️ CSRF Protection
**Status:** ✅ Implemented

- Added Flask-WTF for CSRF token protection
- All POST/PUT/DELETE/PATCH requests now require valid CSRF tokens
- Automatic CSRF token injection in all AJAX requests (fetch & XMLHttpRequest)
- CSRF tokens included in HTML forms via meta tag

**Configuration:** `config.py`
```python
WTF_CSRF_ENABLED = True
WTF_CSRF_TIME_LIMIT = None  # CSRF tokens don't expire
```

### 🔐 Strong Secret Key
**Status:** ✅ Implemented

- Replaced weak default `"super-secret-key-change-in-production"` with auto-generated strong random key
- Uses `secrets.token_hex(32)` to generate 64-character hexadecimal secret
- Supports environment variable override: `SECRET_KEY=your-key`

**Action Required:**
1. Set `SECRET_KEY` environment variable in production
2. Generate strong key: `python3 -c "import secrets; print(secrets.token_hex(32))"`
3. Add to `.env` file (see `.env.example`)

### 📦 File Upload Security
**Status:** ✅ Enhanced

**Existing protections:**
- ✅ Whitelist-based file extension validation (`ALLOWED_EXTENSIONS`)
- ✅ `secure_filename()` prevents path traversal attacks
- ✅ Separate directory for uploads

**New protections:**
- ✅ 50MB max file size limit (`MAX_CONTENT_LENGTH`)

### 🚦 Rate Limiting
**Status:** ✅ Implemented

- Added Flask-Limiter to prevent abuse
- Default limits: 200 requests/day, 50 requests/hour per IP
- Uses Redis for distributed rate limiting
- Protects against DoS attacks and API abuse

**Configuration:** `app.py`
```python
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=app.config['REDIS_URL']
)
```

### 🍪 Session Security
**Status:** ✅ Implemented

- `SESSION_COOKIE_HTTPONLY = True` - Prevents XSS cookie theft
- `SESSION_COOKIE_SAMESITE = 'Lax'` - Prevents CSRF attacks
- `SESSION_COOKIE_SECURE` - Set via environment variable for HTTPS (production)

### 🔒 Security Headers
**Status:** ✅ Implemented

Added security headers to all responses:
- `X-Content-Type-Options: nosniff` - Prevents MIME sniffing
- `X-Frame-Options: SAMEORIGIN` - Prevents clickjacking
- `X-XSS-Protection: 1; mode=block` - Enables browser XSS filter

## Existing Security Protections

### ✅ SQL Injection Protection
- Using SQLAlchemy ORM throughout
- No raw SQL queries found
- Parameterized queries via ORM

### ✅ XSS Protection
- Jinja2 auto-escaping enabled by default
- No use of `render_template_string()` or `Markup()`
- No unsafe HTML rendering

### ✅ Path Traversal Protection
- `secure_filename()` used for all file uploads
- Uploads confined to designated directories
- No direct user control over file paths

## ⚠️ Security Considerations & Recommendations

### 🔴 CRITICAL: No Authentication System
**Current State:** All routes are publicly accessible

**Risk:**
- Anyone can upload/delete images
- Anyone can modify device settings
- Anyone can trigger system updates on devices
- Anyone can access all API endpoints

**Recommendations:**
1. **Implement authentication** (Flask-Login, Flask-Security, or OAuth)
2. **Add authorization** (role-based access control)
3. **Protect admin routes** (device management, settings)
4. **Consider API keys** for device communication

**Example Implementation:**
```python
from flask_login import LoginManager, login_required

# Protect sensitive routes
@app.route('/delete_device/<int:index>', methods=['POST'])
@login_required
def delete_device(index):
    # ... route logic
```

### 🟡 Deployment Security Checklist

#### Production Environment
- [ ] Set `SECRET_KEY` environment variable (64+ characters)
- [ ] Enable HTTPS and set `SESSION_COOKIE_SECURE=True`
- [ ] Configure firewall to restrict access
- [ ] Use reverse proxy (nginx/Apache) with proper headers
- [ ] Enable logging and monitoring
- [ ] Regular security updates for dependencies

#### Docker Security
```dockerfile
# Run as non-root user
USER nobody

# Read-only filesystem where possible
volumes:
  - ./data:/app/data:rw
  - ./images:/app/images:rw
  - ./:/app:ro  # Application code read-only
```

#### Database Security
- [ ] Regular backups of SQLite database
- [ ] Restrict file permissions on `data/mydb.sqlite`
- [ ] Consider PostgreSQL for production

#### Secrets Management
- [ ] Use `.env` file (never commit to git)
- [ ] Or use Docker secrets
- [ ] Or use cloud provider secret managers

### 🟢 Security Best Practices Applied

✅ Principle of least privilege (session cookies, headers)
✅ Defense in depth (multiple security layers)
✅ Secure defaults (auto-generated SECRET_KEY)
✅ Input validation (file extensions, size limits)
✅ Output encoding (Jinja2 auto-escaping)
✅ Rate limiting (prevents abuse)

## Testing Security

### Test CSRF Protection
```bash
# Without CSRF token (should fail with 400)
curl -X POST http://localhost:5001/delete_image/test.jpg

# With CSRF token (should succeed)
# Extract token from HTML and include in request
```

### Test Rate Limiting
```bash
# Send rapid requests (should get 429 after limit)
for i in {1..60}; do curl http://localhost:5001/api/search_images?q=test; done
```

### Test File Upload Limits
```bash
# Try uploading 100MB file (should fail with 413)
dd if=/dev/zero of=large.jpg bs=1M count=100
curl -F "file=@large.jpg" http://localhost:5001/
```

## Security Audit Summary

**Audit Date:** 2025-12-14

**Critical Issues Fixed:** 3
- ✅ Weak SECRET_KEY
- ✅ Missing CSRF protection
- ✅ No rate limiting

**Medium Issues Fixed:** 1
- ✅ No file size limits

**Remaining Issues:** 1
- ⚠️ No authentication/authorization (requires design decision)

**Security Posture:** **Significantly Improved**
- Previous: Multiple critical vulnerabilities
- Current: Core protections in place, authentication needed

## Getting Help

For security issues:
1. Review this documentation
2. Check Flask security best practices: https://flask.palletsprojects.com/en/latest/security/
3. OWASP Top 10: https://owasp.org/www-project-top-ten/
4. Report security vulnerabilities privately (do not create public issues)

## Changelog

### 2025-12-14 - Security Hardening
- Added CSRF protection with Flask-WTF
- Implemented rate limiting with Flask-Limiter
- Strengthened SECRET_KEY generation
- Added security headers (X-Content-Type-Options, X-Frame-Options, X-XSS-Protection)
- Added session cookie security flags
- Implemented 50MB file upload size limit
- Created security documentation
- Added .env.example for secure configuration
