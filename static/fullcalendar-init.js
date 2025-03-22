// FullCalendar initialization script - v6.1.16
document.addEventListener('DOMContentLoaded', function() {
  const calendarEl = document.getElementById('calendar');
  
  if (typeof FullCalendar !== 'undefined') {
    const calendar = new FullCalendar.Calendar(calendarEl, {
      initialView: 'dayGridMonth',
      firstDay: 1,  // Monday
      headerToolbar: {
        left: '',
        center: 'title',
        right: ''
      },
      timeZone: 'local',
      selectable: true,
      editable: true,
      droppable: false,
      eventTimeFormat: {
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
      },
      eventDisplay: 'block',
      eventDidMount: handleEventDidMount,
      eventClick: handleEventClick,
      dateClick: handleDateClick,
      eventDrop: handleEventDrop,
      select: handleDateSelect,
      events: '/schedule/events'
    });
    
    calendar.render();
    
    // Make calendar available globally
    window.calendar = calendar;
  } else {
    console.error('FullCalendar not properly loaded!');
  }
});