<?php
// Check if the form was submitted
if($_SERVER["REQUEST_METHOD"] == "POST"){
    // Check if file was uploaded without errors
    //echo $_FILES["files"]["name"];

    //print_r($_FILES);

    //print_r($_FILES["files"]);
    echo $_FILES["files"]["name"][0];
    if(isset($_FILES["files"])){ 
      //&& $_FILES["files"]["error"] == 0){
        $allowed = array("jpg" => "image/jpg", "jpeg" => "image/jpeg", "gif" => "image/gif", "png" => "image/png");
        $filename = $_FILES["files"]["name"][0];
        $filetype = $_FILES["files"]["type"][0];
        $filesize = $_FILES["files"]["size"][0];
    
        echo "  check this";
        // Verify file extension
        $ext = pathinfo($filename, PATHINFO_EXTENSION);
        if(!array_key_exists($ext, $allowed)) die("Error: Please select a valid file format.");
    
        // Verify file size - 5MB maximum
        $maxsize = 100 * 1024 * 1024;
        if($filesize > $maxsize) die("Error: File size is larger than the allowed limit.");
    
        // Verify MYME type of the file
        if(in_array($filetype, $allowed)){
            // Check whether file exists before uploading it
            if(file_exists("upload/" . $filename)){
                echo $filename . " is already exists.";
            } else{
                move_uploaded_file($_FILES["files"]["tmp_name"][0], "uploads/" . $filename);
                echo "Your file was uploaded successfully.";
            } 
        } else{
            echo "Error: There was a problem uploading your file. Please try again."; 
        }
    } else{
       echo "check that";
       echo $_SERVER["REQUEST_METHOD"];
        echo "Error: " . $_FILES["photo"]["error"];
    }
}
?>
