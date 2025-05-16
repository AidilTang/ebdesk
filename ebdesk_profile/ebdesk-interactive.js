document.addEventListener("DOMContentLoaded", function() {
    // Navigation and Menu Functionality
    const menuToggle = document.querySelector('.menu-toggle');
    const sidebar = document.querySelector('.sidebar');
    const menuItems = document.querySelectorAll('.menu a');
    const submenuParents = document.querySelectorAll('.has-submenu');
    
    // Mobile menu toggle
    menuToggle.addEventListener('click', () => {
        sidebar.classList.toggle('active');
        menuToggle.classList.toggle('active');
    });
    
    // Handle submenu toggles
    submenuParents.forEach(item => {
        item.addEventListener('click', function(e) {
            if (e.target === this.querySelector('a') || e.target === this) {
                e.preventDefault();
                this.classList.toggle('open');
                
                // Close other open submenus
                submenuParents.forEach(otherItem => {
                    if (otherItem !== this && !this.contains(otherItem)) {
                        otherItem.classList.remove('open');
                    }
                });
            }
        });
    });
    
    // Smooth scrolling for anchor links
    menuItems.forEach(item => {
        item.addEventListener('click', function(e) {
            if (this.getAttribute('href').startsWith('#')) {
                e.preventDefault();
                const targetId = this.getAttribute('href');
                if (targetId === '#top') {
                    window.scrollTo({
                        top: 0,
                        behavior: 'smooth'
                    });
                } else {
                    const targetElement = document.querySelector(targetId);
                    if (targetElement) {
                        targetElement.scrollIntoView({
                            behavior: 'smooth'
                        });
                    }
                }
                
                // Close sidebar on mobile after clicking
                if (window.innerWidth < 768) {
                    sidebar.classList.remove('active');
                    menuToggle.classList.remove('active');
                }
            }
        });
    });
    
    // Floating Particles Animation
    const particlesContainer = document.querySelector('.floating-particles');
    
    function createParticles() {
        particlesContainer.innerHTML = '';
        const particleCount = 30;
        
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            
            // Random position, size and opacity
            const size = Math.random() * 10 + 2;
            const posX = Math.random() * 100;
            const posY = Math.random() * 100;
            const opacity = Math.random() * 0.6 + 0.2;
            const duration = Math.random() * 20 + 10;
            const delay = Math.random() * 5;
            
            particle.style.width = `${size}px`;
            particle.style.height = `${size}px`;
            particle.style.left = `${posX}%`;
            particle.style.top = `${posY}%`;
            particle.style.opacity = opacity;
            particle.style.animationDuration = `${duration}s`;
            particle.style.animationDelay = `${delay}s`;
            
            particlesContainer.appendChild(particle);
        }
    }
    
    createParticles();
    
    // Add some CSS for particles
    const style = document.createElement('style');
    style.textContent = `
        .floating-particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
            overflow: hidden;
        }
        
        .particle {
            position: absolute;
            background-color: #3498db;
            border-radius: 50%;
            opacity: 0.2;
            animation-name: float-particle;
            animation-timing-function: ease-in-out;
            animation-iteration-count: infinite;
        }
        
        @keyframes float-particle {
            0% {
                transform: translateY(0) translateX(0);
            }
            25% {
                transform: translateY(-50px) translateX(20px);
            }
            50% {
                transform: translateY(-100px) translateX(-20px);
            }
            75% {
                transform: translateY(-50px) translateX(-40px);
            }
            100% {
                transform: translateY(0) translateX(0);
            }
        }
        
        .menu-toggle {
            display: none;
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            font-size: 24px;
            cursor: pointer;
            background: #2c3e50;
            color: white;
            width: 40px;
            height: 40px;
            text-align: center;
            line-height: 40px;
            border-radius: 5px;
        }
        
        .menu-toggle.active {
            background: #e74c3c;
        }
        
        @media (max-width: 768px) {
            .menu-toggle {
                display: block;
            }
            
            .sidebar {
                transform: translateX(-100%);
                transition: transform 0.3s ease;
            }
            
            .sidebar.active {
                transform: translateX(0);
            }
        }
        
        /* Fade-in animation for sections */
        .fade-in {
            opacity: 0;
            transform: translateY(30px);
            transition: opacity 0.8s ease, transform 0.8s ease;
        }
        
        .fade-in.visible {
            opacity: 1;
            transform: translateY(0);
        }
        
        /* Typed text effect */
        .typed-text::after {
            content: '|';
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
        }
        
        /* Stats counter animation */
        .stat-counter {
            display: inline-block;
            font-weight: bold;
        }
    `;
    document.head.appendChild(style);
    
    // Scroll Animation for Sections
    const sections = document.querySelectorAll('section');
    
    sections.forEach(section => {
        section.classList.add('fade-in');
    });
    
    function checkVisibility() {
        const windowHeight = window.innerHeight;
        
        sections.forEach(section => {
            const sectionTop = section.getBoundingClientRect().top;
            if (sectionTop < windowHeight * 0.85) {
                section.classList.add('visible');
            }
        });
    }
    
    // Initial check
    checkVisibility();
    
    // Check on scroll
    window.addEventListener('scroll', checkVisibility);
    
    // Animated Counter for Stats
    const statsSection = document.querySelector('.stats');
    if (statsSection) {
        const statNumbers = statsSection.querySelectorAll('h3');
        
        function animateCounter(el, target) {
            let count = 0;
            const duration = 2000; // 2 seconds
            const frameDuration = 1000 / 60; // 60fps
            const totalFrames = duration / frameDuration;
            const increment = target / totalFrames;
            
            const counter = setInterval(() => {
                count += increment;
                if (count >= target) {
                    el.textContent = target + '+';
                    clearInterval(counter);
                } else {
                    el.textContent = Math.floor(count) + '+';
                }
            }, frameDuration);
        }
        
        let animated = false;
        
        function checkStatsVisibility() {
            if (!animated && statsSection.getBoundingClientRect().top < window.innerHeight * 0.8) {
                statNumbers.forEach(stat => {
                    const target = parseInt(stat.textContent);
                    stat.textContent = '0';
                    animateCounter(stat, target);
                });
                animated = true;
            }
        }
        
        window.addEventListener('scroll', checkStatsVisibility);
        checkStatsVisibility(); // Check initial visibility
    }
    
    // Typed text effect for header subtitle
    const headerSubtitle = document.querySelector('header h2');
    if (headerSubtitle) {
        const text = headerSubtitle.textContent;
        headerSubtitle.textContent = '';
        headerSubtitle.classList.add('typed-text');
        
        let charIndex = 0;
        const typeInterval = setInterval(() => {
            if (charIndex < text.length) {
                headerSubtitle.textContent += text.charAt(charIndex);
                charIndex++;
            } else {
                clearInterval(typeInterval);
                headerSubtitle.classList.remove('typed-text');
            }
        }, 50);
    }
    
    // Active section highlighting in menu
    function updateActiveMenuItem() {
        let currentSectionId = null;
        const scrollPosition = window.scrollY;
        
        // Find the current section
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.offsetHeight;
            
            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                currentSectionId = section.getAttribute('id');
            }
        });
        
        // If at the top of the page, highlight Home
        if (scrollPosition < 100) {
            currentSectionId = 'top';
        }
        
        // Update active menu item
        menuItems.forEach(item => {
            item.classList.remove('active');
            const href = item.getAttribute('href').substring(1);
            
            if (href === currentSectionId) {
                item.classList.add('active');
            }
        });
    }
    
    window.addEventListener('scroll', updateActiveMenuItem);
    updateActiveMenuItem(); // Initial check
    
    // Initialize contact form with validation and animation
    const contactForm = document.querySelector('.contact-form form');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Simple validation
            let valid = true;
            const inputs = this.querySelectorAll('input, textarea');
            
            inputs.forEach(input => {
                if (!input.value.trim()) {
                    valid = false;
                    input.classList.add('error');
                    
                    // Remove error class after user starts typing
                    input.addEventListener('input', function() {
                        this.classList.remove('error');
                    }, { once: true });
                } else {
                    input.classList.remove('error');
                }
            });
            
            if (valid) {
                // Show success message
                const successMsg = document.createElement('div');
                successMsg.className = 'success-message';
                successMsg.textContent = 'Thank you for your message! We will get back to you soon.';
                
                contactForm.style.opacity = '0';
                setTimeout(() => {
                    contactForm.parentNode.insertBefore(successMsg, contactForm);
                    contactForm.style.display = 'none';
                }, 300);
            }
        });
        
        // Add necessary CSS for form validation
        const formStyle = document.createElement('style');
        formStyle.textContent = `
            .form-input.error, .form-textarea.error {
                border-color: #e74c3c !important;
                animation: shake 0.5s;
            }
            
            @keyframes shake {
                0%, 100% { transform: translateX(0); }
                10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
                20%, 40%, 60%, 80% { transform: translateX(5px); }
            }
            
            .success-message {
                background: #2ecc71;
                color: white;
                padding: 20px;
                border-radius: 5px;
                text-align: center;
                animation: fadeIn 0.5s;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
        `;
        document.head.appendChild(formStyle);
    }
    
    // Create a back-to-top button
    const backToTopBtn = document.createElement('button');
    backToTopBtn.className = 'back-to-top';
    backToTopBtn.innerHTML = '↑';
    document.body.appendChild(backToTopBtn);
    
    // Add styles for back-to-top button
    const backToTopStyle = document.createElement('style');
    backToTopStyle.textContent = `
        .back-to-top {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 40px;
            height: 40px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 50%;
            font-size: 20px;
            cursor: pointer;
            display: flex;
            justify-content: center;
            align-items: center;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s, visibility 0.3s;
            z-index: 1000;
        }
        
        .back-to-top.visible {
            opacity: 1;
            visibility: visible;
        }
        
        .back-to-top:hover {
            background-color: #2980b9;
        }
    `;
    document.head.appendChild(backToTopStyle);
    
    // Show/hide back-to-top button
    function toggleBackToTopButton() {
        if (window.scrollY > 300) {
            backToTopBtn.classList.add('visible');
        } else {
            backToTopBtn.classList.remove('visible');
        }
    }
    
    window.addEventListener('scroll', toggleBackToTopButton);
    
    // Scroll to top when button is clicked
    backToTopBtn.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
    
    // Interactive hover effects for sections
    sections.forEach(section => {
        section.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.005)';
            this.style.transition = 'transform 0.3s ease';
        });
        
        section.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
        });
    });

    // Add interactive hover effects to the stats boxes
    const statBoxes = document.querySelectorAll('.stats > div');
    if (statBoxes.length > 0) {
        statBoxes.forEach(box => {
            box.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-10px)';
                this.style.transition = 'transform 0.3s ease, box-shadow 0.3s ease';
                this.style.boxShadow = '0 10px 20px rgba(0,0,0,0.1)';
            });
            
            box.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0)';
                this.style.boxShadow = 'none';
            });
        });
    }
    
    // Create a custom cursor effect
    const cursor = document.createElement('div');
    cursor.className = 'custom-cursor';
    document.body.appendChild(cursor);
    
    const cursorStyle = document.createElement('style');
    cursorStyle.textContent = `
        .custom-cursor {
            width: 20px;
            height: 20px;
            border: 2px solid #3498db;
            border-radius: 50%;
            position: fixed;
            pointer-events: none;
            transform: translate(-50%, -50%);
            z-index: 9999;
            transition: width 0.2s, height 0.2s, background-color 0.2s;
            mix-blend-mode: difference;
        }
        
        .custom-cursor.active {
            width: 40px;
            height: 40px;
            background-color: rgba(52, 152, 219, 0.2);
        }
        
        @media (max-width: 768px) {
            .custom-cursor {
                display: none;
            }
        }
    `;
    document.head.appendChild(cursorStyle);
    
    document.addEventListener('mousemove', (e) => {
        cursor.style.left = e.clientX + 'px';
        cursor.style.top = e.clientY + 'px';
    });
    
    document.addEventListener('mousedown', () => {
        cursor.classList.add('active');
    });
    
    document.addEventListener('mouseup', () => {
        cursor.classList.remove('active');
    });
    
    // Add interactive elements hover effect
    const interactiveElements = document.querySelectorAll('a, button, .menu-toggle');
    interactiveElements.forEach(el => {
        el.addEventListener('mouseenter', () => {
            cursor.classList.add('active');
        });
        
        el.addEventListener('mouseleave', () => {
            cursor.classList.remove('active');
        });
    });
    
    // Parallax effect for header
    const header = document.querySelector('header');
    if (header) {
        window.addEventListener('scroll', () => {
            const scrollPosition = window.scrollY;
            if (scrollPosition < window.innerHeight) {
                header.style.backgroundPositionY = scrollPosition * 0.5 + 'px';
            }
        });
    }

    // Add a subtle background loading progress bar
    const progressBar = document.createElement('div');
    progressBar.className = 'progress-bar';
    document.body.appendChild(progressBar);
    
    const progressStyle = document.createElement('style');
    progressStyle.textContent = `
        .progress-bar {
            position: fixed;
            top: 0;
            left: 0;
            height: 3px;
            background: linear-gradient(to right, #3498db, #2ecc71, #f1c40f);
            width: 0;
            z-index: 1001;
            transition: width 0.2s;
        }
    `;
    document.head.appendChild(progressStyle);
    
    // Update progress bar based on scroll position
    function updateProgressBar() {
        const windowHeight = document.documentElement.scrollHeight - window.innerHeight;
        const scrolled = (window.scrollY / windowHeight) * 100;
        progressBar.style.width = scrolled + '%';
    }
    
    window.addEventListener('scroll', updateProgressBar);
    updateProgressBar(); // Initial update
    
    // Add a dark mode toggle button
    const darkModeToggle = document.createElement('button');
    darkModeToggle.className = 'dark-mode-toggle';
    darkModeToggle.innerHTML = '🌙';
    darkModeToggle.title = 'Toggle Dark Mode';
    document.body.appendChild(darkModeToggle);
    
    const darkModeStyle = document.createElement('style');
    darkModeStyle.textContent = `
        .dark-mode-toggle {
            position: fixed;
            bottom: 20px;
            left: 20px;
            width: 40px;
            height: 40px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 50%;
            font-size: 16px;
            cursor: pointer;
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            transition: background-color 0.3s;
        }
        
        .dark-mode-toggle:hover {
            background-color: #2980b9;
        }
        
        body.dark-mode {
            background-color: #121212;
            color: #f5f5f5;
        }
        
        body.dark-mode section {
            background-color: #1e1e1e;
        }
        
        body.dark-mode a {
            color: #3498db;
        }
        
        body.dark-mode .sidebar {
            background-color: #1e1e1e;
        }
        
        body.dark-mode .menu a {
            color: #f5f5f5;
        }
        
        body.dark-mode .stats > div {
            background-color: #2c2c2c;
        }
        
        body.dark-mode table {
            background-color: #2c2c2c;
        }
        
        body.dark-mode th, body.dark-mode td {
            border-color: #444;
        }
    `;
    document.head.appendChild(darkModeStyle);
    
    // Check for saved preference
    if (localStorage.getItem('darkMode') === 'true') {
        document.body.classList.add('dark-mode');
        darkModeToggle.innerHTML = '☀️';
    }
    
    darkModeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        
        if (document.body.classList.contains('dark-mode')) {
            localStorage.setItem('darkMode', 'true');
            darkModeToggle.innerHTML = '☀️';
        } else {
            localStorage.setItem('darkMode', 'false');
            darkModeToggle.innerHTML = '🌙';
        }
    });
});
