import React, { useState } from 'react';

const HighlightForm = () => {
  const [image, setImage] = useState('assets/sample_xray.jpg');
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();

    // Fake update results after submission
    setImage('assets/sample_xray_highlighted.jpg');
    if (window.getSelection) {
      setText(window.getSelection().toString());
    }  
  };

  return (
    <div className="d-flex align-items-center justify-content-center" style={{ minHeight: '100vh' }}>
      <div className="card" style={{ maxWidth: '400px' }}>
        <div className="card-body">
          <form onSubmit={handleSubmit}>
            <div className="mb-3 ">
              <label htmlFor="image" className="form-label">XRay Image</label>
              <div className="mb-4 d-flex justify-content-center">
                <img
                  id="selectedImage"
                  src={image}
                  alt="example placeholder"
                  style={{ width: '300px' }}
                />
              </div>
            </div>
            <div className="mb-3">
              <label htmlFor="text" className="form-label">Highlight some text below:</label>
              <p>Radiology Report sample content: your lung is bad, you vape too much bla bla bla</p>
            </div>
            {text && 
              <div className="mb-3">
                <label htmlFor="text" className="form-label">Explained Radiology Report:</label>
                <div class="alert alert-success" role="alert">
                  {text}
                </div> 
              </div> 
            }
            <div className='d-flex justify-content-center'>
              <button type="submit" className="btn btn-primary">Explain</button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default HighlightForm;
