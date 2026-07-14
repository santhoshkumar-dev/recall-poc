import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Button } from "./button";

describe("Button", () => {
  it("renders an accessible button", () => {
    render(<Button>Search</Button>);
    expect(screen.getByRole("button", { name: "Search" })).toBeEnabled();
  });
});
