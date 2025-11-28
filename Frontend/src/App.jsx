import './App.css'
import { useState, useEffect } from 'react'
import Home from './Pages/Home/Home'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import TestAI from './Pages/TestAI/TestAI'
import TestImage from './Pages/TestImage/TestImage'
import TestVideo from './Pages/TestVideo/TestVideo'
import Misbot from './Pages/MisBot/MisBot'
import IntroScreen from './Pages/IntroScreen/IntroScreen'
import ApproachSolution from './Pages/ApproachSolution/ApproachSolution'
import Lenis from 'lenis'



function App() {
  const [showIntro, setShowIntro] = useState(true);

  const handleIntroComplete = () => {
    setShowIntro(false);
  };

  // Initialize Lenis smooth scrolling
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)), // easeOutExpo
      orientation: 'vertical',
      gestureOrientation: 'vertical',
      smoothWheel: true,
      wheelMultiplier: 1,
      smoothTouch: false,
      touchMultiplier: 2,
      infinite: false,
    });

    function raf(time) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }

    requestAnimationFrame(raf);

    // Cleanup
    return () => {
      lenis.destroy();
    };
  }, []);

  const router = createBrowserRouter([
    {
      path: '/',
      element: <> <Home /> </>
    },
    {
      path: '/testai',
      element: <> <TestAI /> </>
    },
    {
      path: '/testimage',
      element: <> <TestImage /> </>
    },
    {
      path: '/testvideo',
      element: <> <TestVideo /> </>
    },
    {
      path: '/misbot',
      element: <> <Misbot /> </>
    },
    {
      path: '/approach',
      element: <> <ApproachSolution /> </>
    },
  ])

  // Show intro screen first, then main app
  if (showIntro) {
    return <IntroScreen onComplete={handleIntroComplete} />;
  }

  return (

    <div className="app-container">
      <RouterProvider router={router} />
    </div>
  );
}

export default App
