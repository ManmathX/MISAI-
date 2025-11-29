import React from 'react'
import logo from '/Images/logo.png'
import { NavLink } from 'react-router'

import './Navbar.css'

const Navbar = () => {
  return (
    <div>

      <div className='navbar-section'>

        <header>
          <div class="nav-container">
            <div class="logo-section"> <img src={logo} class="navlogo" /> </div>

            <nav class="nav-links" >
              <NavLink className="btn" to="/">Home</NavLink>
              <NavLink className="btn" to="/testai">Test Your AI</NavLink>
              <NavLink className="btn" to="/approach">Our Approach</NavLink>
              <NavLink className="btn" to="/testimage">Image Checker </NavLink>
              <NavLink className="btn" to="/misbot">MisBot</NavLink>
            </nav>
          </div>
        </header>


      </div>




    </div>
  )
}

export default Navbar