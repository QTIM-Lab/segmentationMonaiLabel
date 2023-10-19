import React, { useState } from 'react';

const DisplayImgArea = (props) => {

  const [responseData, setResponseData] = useState(null);

  const handleClick = () => {
    // Define the URL and headers
    const url = 'http://0.0.0.0:8000/datastore/image?image=' + props.imgId;
    const headers = {
      'Accept': 'application/json',
    };
  
    // Send a GET request using the fetch API
    fetch(url, {
      method: 'GET',
      headers: headers,
    })
      .then(response => response.blob())
      .then(imageBlob => {
        const imageUrl = URL.createObjectURL(imageBlob);
        props.setImageResponseData(imageUrl);
        props.setInferenceResponseData(null)
      })
      .catch(error => {
        console.error('Error:', error);
      });
  };
  



  return (
    <div>

    <button onClick={handleClick}>Show Image</button>
        {props.imageResponseData && 
            <img src={props.imageResponseData} alt="Fetched Image" />
        }

    </div>
  );
}

export default DisplayImgArea;