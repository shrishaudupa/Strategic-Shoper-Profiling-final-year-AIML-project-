document.addEventListener('DOMContentLoaded', () => {
  const menuToggle = document.querySelector('.toggle');
  const showcase = document.querySelector('.showcase');
  const video = document.getElementById("videoElement");

  // Toggle menu and showcase visibility
  menuToggle.addEventListener('click', () => {
    menuToggle.classList.toggle('active');
    showcase.classList.toggle('active');
  });

  // Function to stop the camera stream
  function stopStream() {
    const stream = video.srcObject;
    if (stream) {
      const tracks = stream.getTracks();
      tracks.forEach(track => track.stop());
      video.srcObject = null;
    }
  }

  // Handle Camera Off button click event
  const cameraOffBtn = document.getElementById('camera-off');
  cameraOffBtn.addEventListener('click', () => {
    // AJAX request to stop the camera feed on the server side
    fetch('homepage/off', {
      method: 'POST',
    })
    .then(response => {
      if (response.ok) {
        console.log('Camera feed stopped successfully');
        stopStream();  // Call stopStream function to stop the video feed on the client side
      } else {
        console.error('Failed to stop camera feed:', response.statusText);
      }
    })
    .catch(error => {
      console.error('Error stopping camera feed:', error);
    });
  });
});
