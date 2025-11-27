import React, { useEffect, useState } from 'react';
import './IntroScreen.css';

function IntroScreen({ onComplete }) {
  const [isExiting, setIsExiting] = useState(false);

  useEffect(() => {
    // Auto-advance to main app after 4.5 seconds
    const timer = setTimeout(() => {
      handleComplete();
    }, 4500);

    return () => clearTimeout(timer);
  }, []);

  const handleComplete = () => {
    setIsExiting(true);
    setTimeout(() => {
      onComplete();
    }, 800); // Wait for exit animation
  };

  return (
    <div className={`intro-screen ${isExiting ? 'exiting' : ''}`}>
      {/* Animated particle background */}
      <div className="particles-container">
        {[...Array(30)].map((_, i) => (
          <div
            key={i}
            className="particle"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 3}s`,
              animationDuration: `${3 + Math.random() * 4}s`
            }}
          />
        ))}
      </div>

      {/* Gradient overlay */}
      <div className="gradient-overlay"></div>

      {/* Main content */}
      <div className="intro-content">
        {/* Logo with reveal animation */}
        <div className="logo-container">
          <h1 className="intro-logo">MISAI</h1>
          <div className="logo-glow"></div>
        </div>

        {/* Tagline with glitch effect */}
        <div className="tagline-container">
          <p className="intro-tagline">AI-Powered Fact Verification</p>
          <p className="intro-tagline glitch" aria-hidden="true">AI-Powered Fact Verification</p>
        </div>

        {/* Loading bar */}
        <div className="loading-bar">
          <div className="loading-progress"></div>
        </div>
      </div>

      {/* Skip button */}
      <button className="skip-button" onClick={handleComplete}>
        Skip Intro →
      </button>
    </div>
  );
}

export default IntroScreen;
