document.getElementById('file-upload').addEventListener('change', function () {
    var fileInput = document.getElementById('file-upload');
    var fileChosen = document.getElementById('file-chosen');
    fileChosen.textContent = fileInput.files[0].name;
});