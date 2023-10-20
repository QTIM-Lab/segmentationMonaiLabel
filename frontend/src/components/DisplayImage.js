import React from 'react';

import UploadImage from './UploadImage';

import { Row, Col } from 'react-bootstrap';

const DisplayImage = props => {

  return (
    <>
        <Row style={{minHeight: '512px'}}>
          {props.imgData && 
              <img src={props.imgData} height={512} width={512} alt="Fetched Image" />
          }
        </Row>
        
        <Row className='my-2'>
          <Col>
            <UploadImage />
          </Col>
        </Row>
    </>
  );
}

export default DisplayImage;