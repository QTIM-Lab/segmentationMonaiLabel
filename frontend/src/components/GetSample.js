import React, { useState } from 'react';

import { Button } from 'react-bootstrap';

const GetSample = props => {
  const [variantName, setVariantName] = useState('primary')

  const handleClick = () => {
      setVariantName('secondary')
      if (props.ctx) {
        props.ctx.clearRect(0, 0, props.canvasRef.current.width, props.canvasRef.current.height);
      }

      // Define the URL and headers
      const url = 'http://localhost:8000/activelearning/last';
      const headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      };
    
      // Define the request body (data)
      const body = JSON.stringify({});
    
      // Send a POST request using the fetch API
      fetch(url, {
        method: 'POST',
        headers: headers,
        body: body,
      })
        .then(response => response.json())
        .then(data => {
          debugger
          const imgId = data.id
          // Define the URL and headers
          const urlImg = 'http://localhost:8000/datastore/image?image=' + imgId;
          const headersImg = {
          'Accept': 'application/json',
          };
      
          // Send a GET request using the fetch API
          fetch(urlImg, {
          method: 'GET',
          headers: headersImg,
          })
          .then(response => response.blob())
          .then(imageBlob => {
              const imageUrl = URL.createObjectURL(imageBlob);
              props.setImageResponseData(imageUrl);
              props.setInferenceResponseData(null);
              props.setImgIdData(data);
              setVariantName('success');
          })
          .catch(error => {
              console.error('Error:', error);
              setVariantName('danger');
          });
        })
        .catch(error => {
          console.error('Error:', error);
          setVariantName('danger');
        });  
  };
    
  return (
    <Button onClick={handleClick} variant={variantName} style={{width: '100%'}}>Get New Sample</Button>
  );
}

export default GetSample;