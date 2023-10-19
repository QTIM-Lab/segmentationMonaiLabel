import React, { useState } from 'react';

const AddLabelArea = (props) => {

    const [responseData, setResponseData] = useState(null);
    const [labelName, setLabelName] = useState('');
    const [labelOptions] = useState([
        "airplane", "automobile", "bird", "cat", "deer", 
        "dog", "frog", "horse", "ship", "truck"
    ]);
  
    const handleClick = () => {
  
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
    const url = `http://0.0.0.0:8000/datastore/label?image=${props.imgId}&tag=final`;
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
      setResponseData(file);
    })
    .catch(error => {
      console.error('Error:', error);
    });
    };
  
    const handleLabelChange = (event) => {
      setLabelName(event.target.value);
    };
  
    return (
      <div className="App">
        <select value={labelName} onChange={handleLabelChange}>
          <option value="">Select a label</option>
          {labelOptions.map(option => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
        <button onClick={handleClick}>Upload Label</button>
        {responseData && <pre>Response: {JSON.stringify(responseData, null, 2)}</pre>}
      </div>
    );
}

export default AddLabelArea;