Dropzone.autoDiscover = false;

function init() {
  let dz = new Dropzone("#dropzone", {
    url: "/",
    maxFiles: 1,
    addRemoveLinks: true,
    dictDefaultMessage: "Test message",
    autoProcessQueue: false,
  });

  dz.on("addedfile", function () {
    if (dz.files[1] != null) {
      dz.removeFile(dz.files[0]);
    }
  });

  dz.on("complete", function (file) {
    let imageData = file.dataURL;

    var url = "https://python-for-data-science-project-end-of.onrender.com/classify_image";

    $.post(
      url,
      {
        image_data: file.dataURL,
        model_name: $("#model_name").val(),
      },
      function (data, status) {
        console.log(data);
        if (!data || !data.svm || data.svm.length == 0) {
          $("#resultHolder").hide();
          $("#divClassTable").hide();
          $("#error").show();
          return;
        }

        // Display SVM results
        displayResults(data.svm, "svm");

        // Display Deep Learning results
        // displayResults(data.dl, "dl");
        dz.removeFile(file);
      }
    );

    function displayResults(modelData, modelName) {
      if (modelData.length > 0) {
        let bestMatch = findBestMatch(modelData);

        if (bestMatch) {
          $("#error").hide();
          $("#resultHolder").show();
          $("#divClassTable").show();

          if ($("#resultHolder").children().length === 0) {
            // Ajoute un élément uniquement si #resultHolder est vide
            $("#resultHolder").append(
              $(`[data-player="${bestMatch.class}"]`).html()
            );
          } else {
            // Supprimer l'élément existant dans #resultHolder
            $("#resultHolder").empty();

            // Ajouter le nouvel élément
            $("#resultHolder").append(
              $(`[data-player="${bestMatch.class}"]`).html()
            );
          }

          // Assuming bestMatch.class_dictionary is the dictionary for the model
          let classDictionary = bestMatch.class_dictionary;

          for (let personName in classDictionary) {
            let index = classDictionary[personName];
            let probabilityScore = bestMatch.class_probability[index];
            let elementName = `#${modelName}_score_${personName}`;
            // arrondir à deux chiffres après la virgule
            $(elementName).html(probabilityScore);
          }
        }
      }
    }

    function findBestMatch(modelData) {
      let match = null;
      let bestScore = -1;

      for (let i = 0; i < modelData.length; ++i) {
        let maxScoreForThisClass = Math.max(...modelData[i].class_probability);
        if (maxScoreForThisClass > bestScore) {
          match = modelData[i];
          bestScore = maxScoreForThisClass;
        }
      }

      return match;
    }
  });

  $("#submitBtn").on("click", function (e) {
    dz.processQueue();
  });
}

$(document).ready(function () {
  console.log("ready!");
  $("#error").hide();
  $("#resultHolder").hide();
  $("#divClassTable").hide();

  init();
});
