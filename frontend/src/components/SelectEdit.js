import React, { useState } from 'react';
import { Form } from 'react-bootstrap';

const SelectEdit = props => {
  
  const handleRadioChange = (event) => {
    props.setSelectedOption(event.target.value);
  };

  return (
    <>
      <h4>Select Region To Segment</h4>
      <Form>
        {['Background', 'Disc', 'Cup'].map((label, index) => (
          <Form.Check
            key={label}
            inline
            label={label}
            type="radio"
            id={`inline-radio-${index}`}
            style={{ color: index === 0 ? "blue" : index === 1 ? "green" : "red", fontSize: "20px" }}
            value={label}
            checked={props.selectedOption === label}
            onChange={handleRadioChange}
          />
        ))}
      </Form>
    </>
  );
}

export default SelectEdit;
