import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import BasicForm from './components/BasicForm';

export default function App() {
  return (
    <Router>
      <main style={{display: 'flex', flexDirection: 'column', height: '100vh'}}> 
        <div style={{flexGrow: 1, overflow: 'auto'}}>
          <Routes>
            <Route path="/" element={<BasicForm />} />
          </Routes>
        </div>

        <div id="disclaimer" className="text-center bg-dark text-light p-2" style={{flexShrink: 0}}>
          <p>By using this service, you agree to waive all rights whatsoever. We are definitely HIPPA compliant!!!</p>
        </div>
      </main>
    </Router>
  );
}