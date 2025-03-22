// Import FullCalendar components
import { Calendar } from '@fullcalendar/core';
import dayGridPlugin from '@fullcalendar/daygrid';
import timeGridPlugin from '@fullcalendar/timegrid';
import interactionPlugin from '@fullcalendar/interaction';

// Make FullCalendar available globally
window.FullCalendarBundle = {
  Calendar,
  plugins: {
    dayGrid: dayGridPlugin,
    timeGrid: timeGridPlugin,
    interaction: interactionPlugin
  }
};

// Initialize calendar if the element exists
document.addEventListener('DOMContentLoaded', function() {
  const calendarEl = document.getElementById('calendar');
  if (calendarEl) {
    const calendar = new Calendar(calendarEl, {
      plugins: [
        dayGridPlugin,
        timeGridPlugin,
        interactionPlugin
      ],
      initialView: 'dayGridMonth',
      headerToolbar: {
        left: 'prev,next today',
        center: 'title',
        right: 'dayGridMonth,timeGridWeek,timeGridDay'
      },
      firstDay: 1, // Monday
      timeZone: 'local',
      editable: true,
      selectable: true
    });
    
    calendar.render();
    
    // Make available globally
    window.calendar = calendar;
  }
});
