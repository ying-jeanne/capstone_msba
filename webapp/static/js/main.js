// Bitcoin Price Prediction - Main JavaScript

// Auto-refresh price every 60 seconds
let priceRefreshInterval;

function startPriceRefresh() {
    priceRefreshInterval = setInterval(async () => {
        const priceElement = document.getElementById('currentPrice');
        if (priceElement) {
            try {
                const response = await fetch('/api/latest-price');
                const data = await response.json();
                if (data.status === 'success') {
                    priceElement.textContent = `$${data.data.price.toFixed(2)}`;

                    // Add pulse animation
                    priceElement.classList.add('price-updated');
                    setTimeout(() => {
                        priceElement.classList.remove('price-updated');
                    }, 1000);
                }
            } catch (error) {
                console.error('Error refreshing price:', error);
            }
        }
    }, 60000); // Every 60 seconds
}

// Manual refresh function
async function refreshPrice(event) {
    const priceElement = document.getElementById('currentPrice');
    const btn = event.target;

    // Disable button and show loading
    btn.disabled = true;
    btn.textContent = '🔄 Refreshing...';

    try {
        const response = await fetch('/api/latest-price');
        const data = await response.json();
        if (data.status === 'success') {
            priceElement.textContent = `$${data.data.price.toFixed(2)}`;

            // Success feedback
            btn.textContent = '✅ Updated!';
            setTimeout(() => {
                btn.textContent = '🔄 Refresh Price';
                btn.disabled = false;
            }, 2000);
        }
    } catch (error) {
        console.error('Error refreshing price:', error);
        btn.textContent = '❌ Error';
        setTimeout(() => {
            btn.textContent = '🔄 Refresh Price';
            btn.disabled = false;
        }, 2000);
    }
}

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add animation class to elements when they come into view
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in');
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe cards and sections
document.addEventListener('DOMContentLoaded', () => {
    // Start price refresh if on live page
    if (document.getElementById('currentPrice')) {
        startPriceRefresh();
    }

    // Observe elements for animation
    const animateElements = document.querySelectorAll('.card, .feature-card, .metric-card, .horizon-card, .insight-card');
    animateElements.forEach(el => observer.observe(el));
});

// Clean up interval on page unload
window.addEventListener('beforeunload', () => {
    if (priceRefreshInterval) {
        clearInterval(priceRefreshInterval);
    }
});

// Add CSS animation class
const style = document.createElement('style');
style.textContent = `
    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .price-updated {
        animation: pulse 0.5s ease-in-out;
    }

    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
`;
document.head.appendChild(style);

// Copy transaction hash to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        // Show tooltip
        const tooltip = document.createElement('div');
        tooltip.textContent = 'Copied!';
        tooltip.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #2ca02c;
            color: white;
            padding: 1rem 2rem;
            border-radius: 6px;
            z-index: 9999;
            animation: fadeOut 2s forwards;
        `;
        document.body.appendChild(tooltip);

        setTimeout(() => {
            tooltip.remove();
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
    });
}

// Add copy functionality to hash links
document.addEventListener('DOMContentLoaded', () => {
    const hashElements = document.querySelectorAll('.hash-value, .hash-link');
    hashElements.forEach(el => {
        el.style.cursor = 'pointer';
        el.title = 'Click to copy';
        el.addEventListener('click', (e) => {
            e.preventDefault();
            const hash = el.textContent.trim().split('...')[0];
            if (hash.startsWith('0x')) {
                copyToClipboard(hash);
            }
        });
    });
});

// Export for use in templates
window.refreshPrice = refreshPrice;
