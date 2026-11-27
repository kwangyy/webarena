---
name: shopping
description: Guide to the online store "One Stop Market" (port 7770), as the logged-in customer: my orders and spending, order status, checkout, product reviews, browsing categories, and which page to end on. Load it for any task that starts on the store.
---

# The online store (One Stop Market, port 7770)

## My account and orders
- On the store, "my" orders, purchases, reviews and account mean the logged-in customer's own pages (My Account, My Orders at /sales/order/history/), not the admin panel's grids, which list every customer.
- Order status (Canceled, refunded, Closed) matters only in some questions. Decide which kind the task is:
  1. Which orders: "my latest order", "the most recent N orders", "the last order", or an order number. Pick the newest orders by date, whatever their status: a Canceled order is still one of them. Then report or add up what those orders contain (items, quantities, totals) as shown, the canceled one included; a canceled order's status is reported as it is (it has no delivery date and will never arrive).
  2. Money the customer spent or paid (how much I spent, total spend in a period). Count only orders that went through: leave out every Canceled order, and any refunded one, since it was never paid for or was paid back.
  3. Counts of successful, fulfilled or completed orders. Leave out Canceled orders and refunded ones.
  4. The task names a status ("completed orders", "pending orders", "cancelled orders", "non-cancelled orders"). Use exactly the orders with that status.

## Ordering
- To order, buy, purchase or reorder something, putting it in the cart is not enough: the task is done only when the order is placed. Add the item to the cart, open the checkout (Proceed to Checkout), keep the saved shipping address and choose a shipping method, keep the default payment method, press the Place Order button, and submit 'Done' only on the confirmation page that says the order was placed and gives its order number.

## Product reviews
- Each review is laid out as its title, its rating, its text and then a "Review by NAME" / "Posted on DATE" line: the name comes after the text it belongs to, not before. A review's reviewer is the name in the "Review by" line below its text, written as the store shows it.

## Browsing and the final page
- To browse or list a category, open that exact category's own page, such as /clothing-shoes-jewelry/men/shoes.html -- not its parent, and not a page with ?cat=<number> added (the sidebar's category filter). Be specific, for example, "Woman clothing" is the Clothing category under Women (/women/clothing.html), not the Women page; "PS4 accessories" is Accessories under PlayStation 4 (/playstation-4/accessories.html), not the PlayStation 4 page. The top menu has several levels: hover over each level in turn until the named category's own link appears and open it, or follow a breadcrumb link above a product of that category. Apply any sort or price filter on that page, and before submitting check that the last part of the URL path names the category the task asked for.
- When a task asks you to show or find something for the user, make the final page one that shows the task's own key words as well as the item: for example, search the store for the problem in the task's words ("jaw bruxism") and end on that search's results page if it lists a fitting item (a page showing "jaw bruxism" and "mouth guard"), rather than on a product page that does not repeat those words.
