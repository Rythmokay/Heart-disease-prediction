document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const resultSection = document.getElementById('result-section');
    const resultMessage = document.getElementById('result-message');
    const resultDetails = document.getElementById('result-details');
    const loadingSpinner = document.getElementById('loading-spinner');

    // Handle form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading spinner
        loadingSpinner.style.display = 'flex';
        
        // Hide previous results if any
        resultSection.style.display = 'none';
        
        // Collect form data
        const formData = {
            age: parseInt(document.getElementById('age').value),
            sex: parseInt(document.getElementById('sex').value),
            cp: parseInt(document.getElementById('cp').value),
            trestbps: parseInt(document.getElementById('trestbps').value),
            chol: parseInt(document.getElementById('chol').value),
            fbs: parseInt(document.getElementById('fbs').value),
            restecg: parseInt(document.getElementById('restecg').value),
            thalach: parseInt(document.getElementById('thalach').value),
            exang: parseInt(document.getElementById('exang').value),
            oldpeak: parseFloat(document.getElementById('oldpeak').value),
            slope: parseInt(document.getElementById('slope').value),
            ca: parseInt(document.getElementById('ca').value),
            thal: parseInt(document.getElementById('thal').value)
        };
        
        // Send data to the server
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            
            // Display results
            resultSection.style.display = 'block';
            
            if (data.success) {
                // Format and display the prediction result
                if (data.prediction === 1) {
                    resultMessage.textContent = 'Heart Disease Detected';
                    resultMessage.className = 'result-positive';
                    resultDetails.innerHTML = `
                        <p>Our model has detected potential signs of heart disease.</p>
                        <p>Confidence: ${(data.probability * 100).toFixed(2)}%</p>
                        <p class="important-note">Please consult with a healthcare professional for a proper diagnosis.</p>
                    `;
                } else {
                    resultMessage.textContent = 'No Heart Disease Detected';
                    resultMessage.className = 'result-negative';
                    resultDetails.innerHTML = `
                        <p>Our model does not detect signs of heart disease based on the provided information.</p>
                        <p>Confidence: ${((1 - data.probability) * 100).toFixed(2)}%</p>
                        <p class="important-note">This is not a medical diagnosis. Regular check-ups are recommended.</p>
                    `;
                }
            } else {
                // Handle errors
                resultMessage.textContent = 'Prediction Error';
                resultMessage.className = '';
                resultDetails.textContent = 'There was an error processing your data. Please try again.';
            }
        })
        .catch(error => {
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            
            // Display error
            resultSection.style.display = 'block';
            resultMessage.textContent = 'Server Error';
            resultMessage.className = '';
            resultDetails.textContent = 'There was a problem connecting to the server. Please try again later.';
            console.error('Error:', error);
        });
    });

    // Reset form and hide results
    document.getElementById('reset-btn').addEventListener('click', function() {
        resultSection.style.display = 'none';
    });
});