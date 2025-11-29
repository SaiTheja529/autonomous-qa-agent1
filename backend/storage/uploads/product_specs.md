E-Shop Checkout Specifications
==============================

Product Catalog
- Astro Headphones (SKU: ASTRO-H1) priced at $120.00
- Nova Phone Case (SKU: NOVA-C1) priced at $25.00
- Lumina Charger (SKU: LUMA-C1) priced at $35.00
- All items are in stock; quantities can be updated in the cart summary.

Discount and Pricing Rules
- Discount code `SAVE15` applies a 15% discount to the cart subtotal when the subtotal is at least $50.00.
- Discount codes are case-insensitive and can only be applied once per checkout session.
- Express shipping costs $10.00; Standard shipping is free.
- Taxes are ignored for this prototype; totals are `subtotal - discount + shipping`.

Form and Validation Rules
- Required fields: Full Name, Email, and Address must be provided before payment.
- Email must be in a valid format (must include `@` and a domain).
- Shipping method is required (Standard or Express).
- Payment method is required (Credit Card or PayPal).
- When validation passes and payment is submitted, the UI must show “Payment Successful!”.

Cart Interactions
- “Add to Cart” buttons push the product into the cart summary with default quantity 1.
- Quantities can be edited directly in the summary; total updates immediately.
- Discount code input should show inline errors for invalid or empty codes.
- Removing shipping selection or payment selection should surface inline errors when attempting payment.

User Feedback
- Inline validation errors should appear directly below the related field.
- Success state should show a green confirmation message “Payment Successful!” above the cart summary.
