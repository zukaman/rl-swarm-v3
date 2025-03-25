import { describe, it, expect, vi, beforeEach } from "vitest"
import { getLeaderboard, getRoundAndStage, getGossip } from "./swarm.api"

describe("getLeaderboard", () => {
	beforeEach(() => {
		vi.clearAllMocks()
	})

	it("should fetch all leaders", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(() =>
				Promise.resolve({
					ok: true,
					status: 200,
					json: () =>
						Promise.resolve({
							leaders: new Array(20).fill(undefined).map((_, i) => {
								return {
									id: `node-${i}`,
									score: 20 - i,
									values: [{ x: 0, y: 20 - i }],
								}
							}),
							total: 20,
						}),
				}),
			),
		)

		const result = await getLeaderboard()
		expect(result.leaders).toHaveLength(20)

		vi.stubGlobal(
			"fetch",
			vi.fn(() =>
				Promise.resolve({
					ok: true,
					status: 200,
					json: () =>
						Promise.resolve({
							leaders: new Array(5).fill(undefined).map((_, i) => {
								return {
									id: `node-${i}`,
									score: 20 - i,
									values: [{ x: 0, y: 20 - i }],
								}
							}),
							total: 5,
						}),
				}),
			),
		)

		const resultShort = await getLeaderboard()
		expect(resultShort.leaders).toHaveLength(5)
	})
})

describe("getRoundAndStage", () => {
	beforeEach(() => {
		vi.clearAllMocks()
	})

	it("should fetch round and stage", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(() =>
				Promise.resolve({
					ok: true,
					status: 200,
					json: () =>
						Promise.resolve({
							round: 1,
							stage: 2,
						}),
				}),
			),
		)

		const result = await getRoundAndStage()
		expect(result).toEqual({
			round: 1,
			stage: 2,
		})
	})

	it("should return default values on error", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(() =>
				Promise.resolve({
					ok: false,
					status: 500,
					statusText: "Internal Server Error",
				}),
			),
		)

		await expect(getRoundAndStage()).rejects.toThrow("could not get round and stage: Failed to fetch round and stage: Internal Server Error")
	})

	it("should handle invalid response data", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(() =>
				Promise.resolve({
					ok: true,
					status: 200,
					json: () =>
						Promise.resolve({
							invalid: "data",
						}),
				}),
			),
		)

		const result = await getRoundAndStage()
		expect(result).toEqual({
			round: -1,
			stage: -1,
		})
	})
})

describe("getGossip", () => {
	beforeEach(() => {
		vi.clearAllMocks()
	})

	it("should fetch gossip messages", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(() =>
				Promise.resolve({
					ok: true,
					status: 200,
					json: () =>
						Promise.resolve({
							messages: [
								{ id: "msg1", message: "test message", node: "node1" },
								{ id: "msg2", message: "another message", node: "node2" },
							],
						}),
				}),
			),
		)

		const result = await getGossip({ since: 1 })
		expect(result.messages).toHaveLength(2)
		expect(result.messages[0].message).toBe("test message")
	})

	it("should handle 5xx errors", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(() =>
				Promise.resolve({
					ok: true,
					status: 500,
					json: () => Promise.resolve({ messages: [] }),
				}),
			),
		)

		await expect(getGossip({ since: 1 })).rejects.toThrow("could not get gossip: internal server error")
	})

	it("should handle fetch errors", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(() =>
				Promise.resolve({
					ok: false,
					status: 400,
					statusText: "Bad Request",
				}),
			),
		)

		await expect(getGossip({ since: 1 })).rejects.toThrow("could not get gossip: failed to fetch gossip: Bad Request")
	})

	it("should handle invalid response data", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(() =>
				Promise.resolve({
					ok: true,
					status: 200,
					json: () =>
						Promise.resolve({
							invalid: "data",
						}),
				}),
			),
		)

		const result = await getGossip({ since: 1 })
		expect(result.messages).toEqual([])
	})

	it("should transform 'gensyn' node to 'INITIAL PEER'", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(() =>
				Promise.resolve({
					ok: true,
					status: 200,
					json: () =>
						Promise.resolve({
							messages: [
								{ id: "msg1", message: "test message", node: "gensyn" },
								{ id: "msg2", message: "another message", node: "Gensyn" },
							],
						}),
				}),
			),
		)

		const result = await getGossip({ since: 1 })
		expect(result.messages[0].node).toBe("INITIAL PEER")
		expect(result.messages[1].node).toBe("INITIAL PEER")
	})
})
