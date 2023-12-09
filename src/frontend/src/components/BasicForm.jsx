import React, { useState } from 'react';

const BasicForm = () => {
  const [imgUpload, setimgUpload] = useState(null);
  const [thumbnail, setThumbnail] = useState(null);
  const [report, setReport] = useState('');
  const [highlight, setHighlight] = useState('');
  const [explaining, setExplaining] = useState(false);

  const handleExplain = async (e) => {
    e.preventDefault();
  
    setExplaining(true);
    setThumbnail('loading.png');
  
    // if user highlight some stuff
    let highlightText;
    if (window.getSelection() && window.getSelection().toString().length > 0) {
      highlightText = window.getSelection().toString();
      setHighlight(highlightText);
    } else {
      highlightText = report;
      setHighlight(report);
    }
  
    // send a POST web request to a place
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          image: imgUpload,
          text: highlightText
        }),
      });
  
      if (response.ok) {
        console.log('Form submitted successfully');
        const responseData = await response.json();
        setThumbnail(responseData);
      } else {
        console.error('Form submission failed');
      }
    } catch (error) {
      console.error('Error submitting form:', error);
    }
  
    setExplaining(false);
  };
  

  const displaySelectedImage = (event, _) => {
    const file = event.target.files[0];

    // Check if a file is selected
    if (file) {
      // Read the selected image and update the state to display the thumbnail
      const reader = new FileReader();
      reader.onloadend = () => {
        setThumbnail(reader.result);
        setimgUpload(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="d-flex align-items-center justify-content-center" style={{ minHeight: '50vh' }}>
      <div className="card" style={{ Width: '400px' }}>
        <div className="card-body">
          <form onSubmit={handleExplain}>
            <div className="mb-3">
              <label htmlFor="image" className="form-label">
                XRay Image File:
              </label>
              <div className="mb-4 d-flex justify-content-center">
                {thumbnail ? (
                  <img
                    id="selectedImage"
                    src={thumbnail}
                    alt="selected thumbnail"
                    style={{ maxWidth: '300px', maxHeight: '300px', width: 'auto', height: 'auto' }}
                  />
                ) : (
                  <img
                    id="selectedImage"
                    src="placeholder.svg"
                    alt="example placeholder"
                    style={{ width: '60px' }}
                  />
                )}
              </div>
              {!highlight && 
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
              }
            </div>
            {!highlight && 
            <div className="mb-3">
              <label htmlFor="report">Enter Your Full Radiology Report:</label>
              <textarea
                className="form-control"
                id="report" rows="3"
                placeholder="Enter report"
                onChange={(e) => setReport(e.target.value)} />
            </div>
            }
            {!highlight &&
            <div className="d-flex justify-content-center">
              <button 
                type="submit" 
                className="btn btn-primary" 
                id='submit' 
                disabled={!(thumbnail && report)} 
              >
                Submit
              </button>
            </div>
            }
            {highlight && 
              <div className="mb-3">
                <label htmlFor="text" className="form-label">Please Highlight Part of Your Radiology Report:</label>
                <div className="alert alert-info" role="alert">
                  {report}
                </div> 
              </div> 
            }
            {highlight && 
              <div className="mb-3">
                <label htmlFor="text" className="form-label">Explained Radiology Report:</label>
                <div className="alert alert-success" role="alert">
                  {highlight}
                </div> 
              </div> 
            }
            {highlight && <div className="d-flex justify-content-center">
              <button type="submit" className="btn btn-primary" id='explain' disabled={!(thumbnail && highlight && !explaining)}>
                {explaining ? "Explaining ..." : "Explain"} 
              </button>
            </div>
            }
          </form>
        </div>
      </div>
    </div>
  );
};

export default BasicForm;
