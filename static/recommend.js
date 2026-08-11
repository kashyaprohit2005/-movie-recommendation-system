$(function() {
  const source = document.getElementById('autoComplete');
  const inputHandler = function(e) {
    if (e.target.value == "") {
      $('.movie-button').attr('disabled', true);
    } else {
      $('.movie-button').attr('disabled', false);
    }
  }
  if (source) {
    source.addEventListener('input', inputHandler);
  }

  $('.movie-button').on('click', function() {
    var title = $('.movie').val();
    if (title == "") {
      $('.results').css('display', 'none');
      $('.fail').css('display', 'block').text('Please enter a movie title!');
    } else {
      load_recommendations(title);
    }
  });
});

function load_recommendations(title) {
  $('.results').css('display', 'none');
  $('.fail').css('display', 'none');
  $("#loader").fadeIn(200);

  $.ajax({
    type: 'POST',
    url: '/get_all_movie_data',
    data: { 'name': title },
    dataType: 'json',
    success: function(response) {
      $("#loader").fadeOut(200);
      if (response.status === 'success') {
        $('.results').html(response.html).fadeIn(400);
        $('#autoComplete').val('');
        $(window).scrollTop(0);
      } else {
        $('.fail').css('display', 'block').text(response.message);
      }
    },
    error: function(xhr) {
      $("#loader").fadeOut(200);
      var err_msg = 'Sorry! The movie you requested is not in our database. Please check the spelling or try another movie.';
      if (xhr.responseJSON && xhr.responseJSON.message) {
        err_msg = xhr.responseJSON.message;
      }
      $('.fail').css('display', 'block').text(err_msg);
    }
  });
}
