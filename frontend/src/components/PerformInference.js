import React, { useState, useEffect } from 'react';

import { Button } from 'react-bootstrap';

import 'bootstrap/dist/css/bootstrap.min.css';

const PeformInference = props => {
    const [variantName, setVariantName] = useState('primary')

    useEffect(() => {
        setVariantName('primary')
    }, [props.imgId]);

    const handleClick = () => {
        setVariantName('secondary')
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
          setVariantName('success')
          console.log('hello?: ', imageUrl)
        })
        .catch(error => {
            setVariantName('danger')
          console.error('Error:', error);
        });
    };

    return (
        <Button onClick={handleClick} disabled={!props.imgId} variant={variantName} style={{width: '100%'}}>
            Perform Inference
        </Button>
    );
}

export default PeformInference;
