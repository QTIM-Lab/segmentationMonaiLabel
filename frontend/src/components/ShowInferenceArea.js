import React, { useState } from 'react';

import { Navbar, Nav } from 'react-bootstrap';

import 'bootstrap/dist/css/bootstrap.min.css';

const ShowInferenceArea = (props) => {
  
    const handleClick = () => {  
      // Define the URL and headers
      const url = 'http://localhost:8000/infer/SegformerBundle?image=' + props.imgId + '&output=image';
      const headers = {
        'Accept': 'application/json',
      };
  
      // Create a new FormData object to handle multipart/form-data
      const formData = new FormData();
  
      // Add the parameters, empty file field, and label (if needed) to the FormData
      formData.append('params', JSON.stringify({}));
      formData.append('file', '');
      formData.append('label', '');
  
      // Send a POST request with the FormData using the fetch API
      fetch(url, {
        method: 'POST',
        headers: headers,
        body: formData,
      })
      .then(response => response.blob())
      .then(imageBlob => {
        const imageUrl = URL.createObjectURL(imageBlob);
        props.setInferenceResponseData(imageUrl);
        console.log('hello?: ', imageUrl)
      })
      .catch(error => {
        console.error('Error:', error);
      });
  };
  

  return (
    <div>

    <button onClick={handleClick}>Show Inference Result</button>
        {props.inferenceResponseData && 
            <img src={props.inferenceResponseData} alt="Fetched Image" />
        }

    </div>
  );
}

export default ShowInferenceArea;
