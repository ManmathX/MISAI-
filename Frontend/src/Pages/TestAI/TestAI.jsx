import React from 'react'
import './TestAI.css'
import Navbar from '../Navbar/Navbar'

const TestAI = () => {
  return (
    <>
      <Navbar />
      <div className="testai-container">
        <iframe
          src="https://dashboard-2-jpcx.onrender.com/"
          title="AI Testing Dashboard"
          style={{
            width: '100%',
            height: 'calc(100vh - 80px)',
            border: 'none',
            borderRadius: '8px'
          }}
        />
      </div>
    </>
  )
}

export default TestAI
