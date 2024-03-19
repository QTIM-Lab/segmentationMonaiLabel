import React, { useState, useEffect } from 'react';

import { Button } from 'react-bootstrap';

import 'bootstrap/dist/css/bootstrap.min.css';

const SaveLabel = props => {

    const [variantName, setVariantName] = useState('primary');

    useEffect(() => {
        setVariantName('primary')
    }, [props.imgId]);
  
    const handleClick = () => {
        setVariantName('secondary')

        // Get the canvas element and convert it to a data URL with the "image/png" format.
        const canvas = props.canvasRef.current;
        const imageData = canvas.toDataURL("image/png");
    
        // Check if there's image data to upload.
        if (!imageData) {
            alert('No image data to upload.');
            return;
        }

        console.log("data: ", imageData)
    
        // Define the URL and headers
        const url = `http://localhost:8000/datastore/label?image=${props.imgId}&tag=final`;
        const headers = {
            'Accept': 'application/json',
        };
    
        // Create a new FormData object to handle multipart/form-data
        const formData = new FormData();

        formData.append('params', JSON.stringify({}));
    
        // Add the image data as a blob with a "image/png" type
        const formFileName = `${props.imgId}.png`;

        const byteCharacters = atob(imageData.split(',')[1]);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const imageBlob = new Blob([byteArray], { type: 'image/png' });
        formData.append('label', imageBlob, formFileName);
    
        // Send a PUT request with the FormData using the fetch API
        fetch(url, {
            method: 'PUT',
            headers: headers,
            body: formData,
        })
        .then(response => response.json())
        .then(file => {
            // setResponseData(file);
            setVariantName('success')
        })
        .catch(error => {
            console.error('Error:', error);
            setVariantName('danger')
        });
    };
  
    return (
      <Button onClick={handleClick} variant={variantName} disabled={!(props.imgId && props.canvasRef)} style={{width: '100%'}}>Save Label</Button>
    );
}

export default SaveLabel;
