import React, { useState } from 'react';

const AddImgArea = () => {

    const [responseData, setResponseData] = useState(null);
    const [selectedFile, setSelectedFile] = useState(null);

    const handleClick = () => {
        if (!selectedFile) {
            alert("Please select a file before uploading.");
            return;
        }

        // Define the URL and headers
        const url = 'http://0.0.0.0:8000/datastore/image?image=' + selectedFile['name'].split('.png')[0];
        const headers = {
            'Accept': 'application/json',
        };

        // Create a new FormData object to handle multipart/form-data
        const formData = new FormData();

        // Add the parameters and the selected file to the FormData
        formData.append('params', JSON.stringify({}));
        formData.append('file', selectedFile);

        // Send a PUT request with the FormData using the fetch API
        fetch(url, {
            method: 'PUT',
            headers: headers,
            body: formData,
        })
            .then(response => response.json())
            .then(data => {
            setResponseData(data);
            })
            .catch(error => {
            console.error('Error:', error);
            });
        };

        const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    return (
    <div className="App">
        <input type="file" onChange={handleFileChange} />
        <button onClick={handleClick}>Upload Image</button>
        {responseData && <pre>Response: {JSON.stringify(responseData, null, 2)}</pre>}
    </div>
    );
}

export default AddImgArea;