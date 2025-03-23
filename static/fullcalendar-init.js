// FullCalendar initialization script - v6.1.16
document.addEventListener('DOMContentLoaded', function() {
  const calendarEl = document.getElementById('calendar');
  
  if (typeof FullCalendar !== 'undefined') {
    const calendar = new FullCalendar.Calendar(calendarEl, {
      initialView: 'timeGridWeek',
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
      events: '/schedule/events',
      // Set time grid to start at 6:00 AM
      slotMinTime: '06:00:00',
      scrollTime: '06:00:00',
      // Allow scrolling to earlier times if needed
      slotLabelInterval: '01:00:00',
      allDaySlot: true
    });
    
    calendar.render();
    
    // Make calendar available globally
    window.calendar = calendar;
  } else {
    console.error('FullCalendar not properly loaded!');
  }
});