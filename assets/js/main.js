import { Calendar } from '@fullcalendar/core';
import timeGridPlugin from '@fullcalendar/timegrid';
import '@fullcalendar/core/main.css';
import '@fullcalendar/timegrid/main.css';

document.addEventListener('DOMContentLoaded', function() {
  var calendarEl = document.getElementById('calendar');
  var calendar = new Calendar(calendarEl, {
    plugins: [ timeGridPlugin ],
    initialView: 'timeGridWeek',
    firstDay: 1, // Week starts on Monday
    nowIndicator: true,
    headerToolbar: {
      left: 'prev,next today',
      center: 'title',
      right: 'timeGridWeek,timeGridDay'
    },
    events: '/schedule/events',
    dateClick: function(info) {
      // Open your custom modal with pre-filled date/time (similar to your current code)
      var dtLocal = new Date(info.date);
      var isoStr = dtLocal.toISOString().substring(0,16);
      document.getElementById('eventDate').value = isoStr;
      openEventModal();
    }
  });
  calendar.render();
});
