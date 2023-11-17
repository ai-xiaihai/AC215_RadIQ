import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import BasicForm from './components/BasicForm';

export default function App() {
  return (
    <Router>
      <main>
        <Routes>
          <Route path="/" element={<BasicForm />} />
        </Routes>

        <div id="disclaimer" className="fixed-bottom text-center bg-dark text-light p-2">
          <p>By using this service, you agree to waive all rights whatsoever. We are definitely HIPPA compliant!!!</p>
        </div>
      </main>
    </Router>
  );
}