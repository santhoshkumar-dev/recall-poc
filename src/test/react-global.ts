import React from "react";
import "@testing-library/jest-dom/vitest";

// Vitest 2 defaults to the classic JSX transform for standalone test files.
// Expose React for that transform while the Next.js application uses automatic JSX.
Object.assign(globalThis, { React });
