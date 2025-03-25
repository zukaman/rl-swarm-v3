import { z } from "zod"

const leaderboardResponseSchema = z.object({
	leaders: z.array(
		z.object({
			id: z.string(),
			values: z.array(z.object({ x: z.number(), y: z.number() })),
			score: z.number(),
		}),
	),
	total: z.number(),
})

export type LeaderboardResponse = z.infer<typeof leaderboardResponseSchema>

type GossipRequest = {
	since: number
}

const gossipResponseSchema = z.object({
	messages: z.array(
		z.object({
			id: z.string(),
			message: z.string(),
			node: z.string(),
		}),
	),
})
export type GossipResponse = z.infer<typeof gossipResponseSchema>

const roundAndStageResponseSchema = z.object({
	round: z.number(),
	stage: z.number(),
})

export type RoundAndStageResponse = z.infer<typeof roundAndStageResponseSchema>

export async function getRoundAndStage(): Promise<RoundAndStageResponse> {
	try {
		const res = await fetch("/api/round_and_stage")
		if (!res.ok) {
			throw new Error(`Failed to fetch round and stage: ${res.statusText}`)
		}

		const json = await res.json()
		return roundAndStageResponseSchema.parse(json)
	} catch (e) {
		if (e instanceof z.ZodError) {
			console.warn("zod error fetching round and stage details. returning empty round and stage response.", e)
			return {
				round: -1,
				stage: -1,
			}
		} else if (e instanceof Error) {
			console.error("error fetching round and stage details", e)
			throw new Error(`could not get round and stage: ${e.message}`)
		} else {
			throw new Error("could not get round and stage")
		}
	}
}

export async function getLeaderboard(): Promise<LeaderboardResponse> {
	try {
		const res = await fetch(`/api/leaderboard`)
		if (!res.ok) {
			throw new Error(`Failed to fetch leaderboard: ${res.statusText}`)
		}

		const json = await res.json()
		const result = leaderboardResponseSchema.parse(json)

		result.leaders.forEach((leader) => {
			leader.score = parseFloat(leader.score.toFixed(2))
			if (leader.id.toLowerCase() === "gensyn") {
				leader.id = "INITIAL PEER"
			}
			leader.values = []
		})

		return result
	} catch (e) {
		if (e instanceof z.ZodError) {
			console.warn("zod error fetching leaderboard details. returning empty leaderboard response.", e)
			return {
				leaders: [],
				total: 0,
			}
		} else if (e instanceof Error) {
			console.error("error fetching leaderboard details", e)
			throw new Error(`could not get leaderboard: ${e.message}`)
		} else {
			throw new Error("could not get leaderboard")
		}
	}
}

export async function getGossip(req: GossipRequest): Promise<GossipResponse> {
	try {
		const res = await fetch(`/api/gossip?since_round=${req.since}`)

		if (!res.ok) {
			throw new Error(`failed to fetch gossip: ${res.statusText}`)
		}

		const json = await res.json()

		if (res.status > 499) {
			console.error("5xx error fetching gossip")
			throw new Error("could not get gossip: internal server error")
		}

		const gres = gossipResponseSchema.parse(json)

		gres.messages.forEach((message) => {
			if (message.node.toLocaleLowerCase() === "gensyn") {
				message.node = "INITIAL PEER"
			}
		})

		return gres
	} catch (e) {
		if (e instanceof z.ZodError) {
			console.warn("zod error fetching gossip details. returning empty gossip response.", e)
			return {
				messages: [],
			}
		} else if (e instanceof Error) {
			console.error("error fetching gossip details", e)
			throw new Error(`could not get gossip: ${e.message}`)
		} else {
			throw new Error("could not get gossip")
		}
	}
}
