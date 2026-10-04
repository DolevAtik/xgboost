/**
 * Where "Get in touch" messages go. The form opens the visitor's mail client with the
 * message filled in; nothing is sent through a server.
 *
 * Set `email` to the team's address. While it is empty, the form explains that no
 * address is configured instead of failing silently.
 */
export const CONTACT = {
  email: "",
  subject: "Drive Foresight — investor inquiry",
};

/** Opens the contact dialog from anywhere on the page. */
export const openContact = () => window.dispatchEvent(new CustomEvent("open-contact"));
