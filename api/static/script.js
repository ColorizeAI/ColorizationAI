// Cookie Consent Script
document.addEventListener('DOMContentLoaded', function() {
    if (!localStorage.getItem('cookieConsent')) {
        // Show the cookie consent popup if not accepted
        document.getElementById('cookie-consent').style.display = 'block';
    }
});

function acceptCookies() {
    // Set a flag in localStorage to remember user's choice
    localStorage.setItem('cookieConsent', 'accepted');
    // Hide the cookie consent popup
    document.getElementById('cookie-consent').style.display = 'none';
}