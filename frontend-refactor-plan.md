# InkyDocker Frontend Refactoring Plan

## Analysis of Current State:

1.  **Technology:** The frontend uses Flask/Jinja2, Bootstrap 5.3 (via CDN), FontAwesome, Cropper.js, and likely FullCalendar.
2.  **CSS:**
    *   `style.css` contains significant overlap with Bootstrap 5 utilities (spacing, cards) and components (navbar).
    *   Excessive use of `!important` suggests specificity issues or overrides that need cleaning up.
    *   Custom modal styles (`.custom-modal`, `.lightbox-modal`) exist alongside Bootstrap's modal system.
    *   Inline styles are present in `base.html`.
    *   `notifications.css` provides custom styling for Bootstrap Toasts and an Offcanvas component used as a notification center.
3.  **JavaScript:**
    *   Files are somewhat scattered (`static/`, `static/js/`).
    *   Inline JS exists in `base.html` for lazy loading and toast container setup.
    *   `notifications.js` handles polling for notifications, displaying toasts, and managing the notification center (Offcanvas).
    *   `live-notifications.js` attempts to provide live feedback for operations like uploads and sending images by intercepting actions and creating/updating notifications. However, the implementation for *updating* existing toasts seems incomplete, and it doesn't explicitly use spinners as requested, relying instead on progress percentages in messages.
4.  **Templates:** Standard Jinja2 templates extending `base.html`. They likely use a mix of Bootstrap classes and the custom classes defined in `style.css`.

## Refactoring Goals:

*   Simplify and standardize CSS by fully leveraging Bootstrap 5.
*   Improve JavaScript organization and modularity.
*   Implement a robust and clear live notification system using Bootstrap Toasts, incorporating spinners for ongoing tasks without clear progress percentages.
*   Ensure all existing functionality is preserved.
*   Remove redundant or unused code.

## Proposed Refactoring Plan:

1.  **CSS Consolidation & Cleanup:**
    *   **Objective:** Reduce custom CSS, eliminate redundancy with Bootstrap, and fix specificity issues.
    *   **Tasks:**
        *   Audit `style.css`: Identify and remove custom classes/styles that duplicate Bootstrap 5 utilities (e.g., `.mb-1`, `.p-2`, custom `.card` styles). Update templates to use the corresponding Bootstrap classes.
        *   Remove the unused custom navbar CSS from `style.css`.
        *   Refactor rules using `!important` to use proper CSS specificity or adjust HTML structure where needed.
        *   Migrate necessary inline styles from `base.html` to a dedicated custom CSS file (e.g., `custom.css`).
        *   Review `notifications.css` to ensure it complements Bootstrap styles effectively without unnecessary overrides. Consider merging it into the main `custom.css` if appropriate.
    *   **Expected Outcome:** A lean CSS setup primarily using Bootstrap, with minimal, well-organized custom styles.

2.  **JavaScript Organization & Enhancement:**
    *   **Objective:** Structure JS logically, eliminate inline scripts, and implement the desired live notification behavior.
    *   **Tasks:**
        *   Create a primary application JS file (e.g., `app.js`) or utilize the existing Webpack setup (`webpack.config.js`) more effectively.
        *   Move inline JavaScript from `base.html` (lazy loading, toast container creation) into `app.js` or an initialization module.
        *   Merge `notifications.js` and `live-notifications.js` into a single, cohesive notification module within `app.js` or as a separate `notifications.js` module imported by `app.js`.
        *   Refactor the live notification logic:
            *   Implement a clear mechanism to track active background tasks (uploads, image sends, etc.) and their associated notification toasts using unique IDs.
            *   Create a function (e.g., `updateNotificationToast(id, message, showSpinner, progress)`) that reliably finds an existing toast by its ID and updates its content (message, spinner visibility, progress bar).
            *   Use Bootstrap Spinners (`<span class="spinner-border spinner-border-sm"></span>`) within the toast body for tasks like "Sending image..." where progress percentage isn't available. Hide the spinner and update the message upon completion (success/error).
            *   For uploads, continue using the progress updates but ensure they modify the *existing* notification toast.
        *   Review `gallery.js`, `device_crop.js`, `settings.js`, `fullcalendar-init.js` for consistency and potential integration into a modular structure.
    *   **Expected Outcome:** Organized, modular JavaScript; functional live notifications using toasts with spinners and progress updates.

3.  **Template Adjustments:**
    *   **Objective:** Ensure templates consistently use Bootstrap 5 conventions and components.
    *   **Tasks:**
        *   Update all templates (`index.html`, `settings.html`, etc.) to replace removed custom CSS classes with standard Bootstrap 5 classes.
        *   Standardize modal usage: Replace custom modal implementations (`.custom-modal`, `.lightbox-modal`) with Bootstrap 5 modals, updating associated JavaScript triggers and event handlers.
        *   Verify the Bootstrap navbar in `base.html` works correctly without the removed custom CSS.
    *   **Expected Outcome:** Clean templates adhering to Bootstrap 5 standards.

4.  **Code Removal:**
    *   **Objective:** Eliminate dead code.
    *   **Tasks:**
        *   Unused CSS identified during Step 1 will be removed.
        *   After refactoring, perform a final review (manual or using browser tools) to catch any remaining unused CSS or JavaScript functions.
    *   **Expected Outcome:** A smaller, more maintainable codebase.

## Visual Plan (Mermaid):

```mermaid
graph TD
    A[Start Frontend Refactor] --> B(CSS Cleanup);
    B --> B1(Remove Bootstrap Duplicates in style.css);
    B --> B2(Remove Unused Navbar CSS);
    B --> B3(Refactor/Remove !important);
    B --> B4(Move Inline Styles to custom.css);
    B --> B5(Review/Integrate notifications.css);
    B --> C(JS Organization & Enhancement);
    C --> C1(Consolidate JS Files / Use Webpack);
    C --> C2(Move Inline JS from base.html);
    C --> C3(Merge & Refactor Notification Logic);
    C3 --> C3a(Implement Reliable Toast Updates by ID);
    C3 --> C3b(Add Bootstrap Spinners to Toasts);
    C3 --> C3c(Refine Upload Progress Toasts);
    C --> C4(Review Other JS Files);
    B --> D(Template Adjustments);
    C --> D;
    D --> D1(Replace Custom Classes with Bootstrap Classes);
    D --> D2(Standardize on Bootstrap Modals);
    D --> D3(Verify Navbar & Components);
    D --> E(Remove Unused Code);
    E --> F(Final Review & Testing);
    F --> G(Refactor Complete);

    subgraph Live Notifications
        C3a; C3b; C3c; B5;
    end