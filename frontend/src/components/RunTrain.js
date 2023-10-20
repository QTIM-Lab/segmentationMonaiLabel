import React, { useState, useEffect } from 'react';

import { Button } from 'react-bootstrap';

import 'bootstrap/dist/css/bootstrap.min.css';

const RunTrain = props => {

    const [variantName, setVariantName] = useState('primary');
    const [isLoading, setIsLoading] = useState(false);
    const [isDone, setIsDone] = useState(false);

    // useEffect(() => {
    //     setVariantName('primary')
    // }, [props.imgId]);

    useEffect(() => {
        // fetchData(); // Initial data fetch
        const intervalId = setInterval(() => {
            fetch('http://34.91.126.112:8000/train/?all=false&check_if_running=false', {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
            },
            })
            .then(response => response.json())
            .then(data => {
            if (data.status === 'DONE' && isLoading) {
                setIsLoading(false);
                setIsDone(true);
                setVariantName('success')
            }
            })
            .catch(error => {
            console.error('Error fetching data:', error);
            });
          
        }, 3000);
    
        return () => {
          clearInterval(intervalId); // Clear the interval when the component unmounts
        };
    }, [isLoading]);

    // const fetchData = () => {
    //     fetch('http://34.91.126.112:8000/train/?all=false&check_if_running=false', {
    //       method: 'GET',
    //       headers: {
    //         'Accept': 'application/json',
    //       },
    //     })
    //     .then(response => response.json())
    //     .then(data => {
    //       if (data.status === 'DONE' && isLoading) {
    //         setIsLoading(false);
    //         setIsDone(true);
    //         setVariantName('success')
    //       } else {
    //         console.log(isLoading)
    //       }
    //     })
    //     .catch(error => {
    //       console.error('Error fetching data:', error);
    //     });
    // };

    const handleClick = () => {
        setVariantName('secondary');
        setIsLoading(true);
        setIsDone(false);

      // Define the URL and headers
      const url = 'http://34.91.126.112:8000/train/SegformerBundle?run_sync=false&enqueue=false';
      const headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      };
  
      // Define the POST request body data
      const postData = {};
  
      // Send a POST request using the fetch API
      fetch(url, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(postData),
      })
        .then(response => response.json())
        .then(data => {
            console.log(data)
            // setVariantName('secondary')
        //   setResponseData(data);
        })
        .catch(error => {
          console.error('Error:', error);
          setVariantName('danger')
        });
    };
  
    return (
      <Button onClick={handleClick} variant={variantName} style={{width: '100%'}}>
        Train on Labels &nbsp;
        {(isLoading) ? (
        <div className="spinner">
          <div className="spinner-inner"></div>
            </div>
        ) : (isDone) ? (
            <div className="check-mark">✔</div>
        ) : null}
    </Button>
    );
}

export default RunTrain;
