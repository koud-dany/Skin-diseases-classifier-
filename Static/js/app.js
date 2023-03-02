$("#file-picker").change(function(){

    var input = document.getElementById('file-picker');

    for (var i=0; i<input.files.length; i++)
    {
        var ext= input.files[i].name.substring(input.files[i].name.lastIndexOf('.')+1).toLowerCase()

        if ((ext == 'jpg') || (ext == 'png'))
        {
            $("#msg").text("Files are supported")
        }
        else
        {
            $("#msg").text("Files are NOT supported")
            document.getElementById("file-picker").value ="";
        }

    }


} );