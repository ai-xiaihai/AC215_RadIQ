import React, { useState } from 'react';

const BasicForm = () => {
  const [image, setImage] = useState('');
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();

    // Add your logic for handling the form submission here
    console.log('Image:', image);
    console.log('Text:', text);

    // Reset the form fields after submission
    setImage('');
    setText('');
  };

  return (
    <div className="card mx-auto" style={{ maxWidth: '400px' }}>
      <div className="card-body">
        <form onSubmit={handleSubmit}>
          <div className="mb-3 ">
            <label htmlFor="image" className="form-label">XRay Image</label>
            <div className="mb-4 d-flex justify-content-center">
              <img
                id="selectedImage"
                src="https://mdbootstrap.com/img/Photos/Others/placeholder.jpg"
                alt="example placeholder"
                style={{ width: '300px' }}
              />
            </div>
            <div className="d-flex justify-content-center">
              <div className="btn btn-secondary btn-rounded">
                <label className="form-label text-white m-1" htmlFor="customFile1">
                  Choose file
                </label>
                <input
                  type="file"
                  className="form-control d-none"
                  id="customFile1"
                  onChange={(event) => displaySelectedImage(event, 'selectedImage')}
                />
              </div>
            </div>
          </div>
          <div className="mb-3">
            <label htmlFor="text" className="form-label">Radiology Report:</label>
            <input
              type="text"
              className="form-control"
              id="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter text"
            />
          </div>
          <div className='d-flex justify-content-center'>
            <button type="submit" className="btn btn-primary">Explain</button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default BasicForm;
