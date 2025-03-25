import { children, createEffect, createMemo, JSXElement, onCleanup, onMount, ParentProps, Show } from "solid-js"
import * as d3 from "d3"
import Modal from "./Modal"
import { LeaderboardResponse } from "./swarm.api"
import { useSwarm } from "./SwarmContext"
import { createSignal } from "solid-js"

export default function Swarm() {
	const ctx = useSwarm()

	// A derived value based on the leaderboard data.
	// If no leaders are coming in the response, then it's the signal of a new round.
	const showNewRoundModal = createMemo(() => {
		const leaders = ctx.leaders()?.leaders ?? []
		return leaders.length <= 0
	})

	return (
		<main class="tracking-wider !font-normal !antialiased max-w-[876px] p-2 md:pt-12 md:pl-12 md:pr-0 flex flex-col justify-between md:min-h-screen ml-auto mr-auto">
			<Show when={showNewRoundModal()}>
				<Modal data-testid="new-stage-modal" message="&lt; Starting new stage... &gt;" />
			</Show>
			<header>
				<h1 class="uppercase text-2xl tracking-[0.25em] mb-4">Gensyn: RL Swarm Client Interface</h1>
				<Banner>
					<p class="uppercase leading-[calc(1em+8px)] mt-4 mb-4">
						<mark class="p-[3px]">A peer-to-peer system for collaborative reinforcement learning over the internet, running on consumer hardware.</mark>
					</p>
				</Banner>
			</header>

			<article class="flex flex-grow md:h-0 md:min-h-[600px] md:max-h-full gap-4 flex-col md:flex-row mb-12 mt-8">
				<LeaderboardSection leaders={ctx.leaders() ?? undefined} currentRound={ctx.currentRound()} currentStage={ctx.currentStage()} />
				<GossipSection messages={ctx.gossipMessages()} />
			</article>

			<Banner>
				<footer class="flex items-center uppercase mt-8 mb-8">
					<div class="flex-1">
						<a href="https://github.com/gensyn-ai/rl-swarm">
							<mark>Join swarm</mark>
						</a>
					</div>
					<div class="flex items-center">
						<a class="flex items-center" href="https://gensyn.ai" target="_blank" rel="noopener noreferrer">
							<img class="h-[50px]" src="/images/logo.gif" alt="A spinning gensyn logo" />
							<img class="h-[30px]" src="/images/gen-logotype-dark-rgb.svg" alt="The gensyn name" />
						</a>
					</div>
					<div class="flex items-center ml-8">gensyn &copy;2025</div>
				</footer>
			</Banner>
		</main>
	)
}

function SectionHeader(props: { title: string; tooltip?: string | JSXElement }) {
	return (
		<header class="flex items-center mb-4">
			<div class="flex-1">
				<mark class="uppercase">{props.title}</mark>
			</div>

			{props.tooltip ? <Tooltip message={props.tooltip} /> : null}
		</header>
	)
}

function Tooltip(props: { message: string | JSXElement }) {
	let detailsRef: HTMLDetailsElement | undefined = undefined

	const close = () => {
		if (detailsRef === undefined) {
			return
		}

		;(detailsRef as HTMLDetailsElement).removeAttribute("open")
	}

	onMount(() => {
		document.addEventListener("click", close)
	})

	onCleanup(() => {
		document.removeEventListener("click", close)
	})

	return (
		<div class="relative">
			<details class="group" ref={detailsRef}>
				<summary class="cursor-pointer list-none flex-none tracking-tightest text-xs tracking-[-0.25em] no-marker">[+]</summary>
				<div class="fixed inset-0 bg-black/50 flex items-center justify-center">
					<div class="max-w-[80vw] md:max-w-[33vw] px-8 py-8 bg-[#fcc6be] text-[#2A0D04] textsm border border-[#2A0D04] max-h-[90vh] md:max-h-[50vh] overflow-y-auto overflow-x-hidden">
						<div class="justify-end w-full text-right mb-2">
							<button class="cursor-pointer" onClick={close}>
								<mark>[&times; Close]</mark>
							</button>
						</div>
						<div onClick={(e) => e.stopPropagation()}>{props.message}</div>
					</div>
				</div>
			</details>
		</div>
	)
}

function Banner(props: ParentProps) {
	const resolvedChildren = children(() => props.children)
	return (
		<>
			<hr class="h-4 bg-[url('/images/line-asterisk.svg')] bg-repeat-x bg-left border-0 flex-shrink-0" />
			{resolvedChildren()}
			<hr class="h-4 bg-[url('/images/line-oblique.svg')] bg-repeat-x bg-left border-0 flex-shrink-0" />
		</>
	)
}

function NodeMessage(props: { id: string; message: string }) {
	const reAnswer = new RegExp(/Answer:.+$/)
	const match = props.message.match(reAnswer)

	let mainText = props.message
	let answer = ""

	if (match) {
		mainText = mainText.slice(0, props.message.length - match[0].length)
		answer = match[0]
	}

	return (
		<p class="uppercase">
			<span class="text-gensyn-green">[{props.id}]</span> {mainText} <strong>{answer}</strong>
		</p>
	)
}

function GossipSection(props: { messages: { id: string; message: string; node: string }[] }) {
	let containerRef: HTMLDivElement | undefined

	const scrollToBottom = () => {
		if (containerRef) {
			// Use requestAnimationFrame to ensure DOM has updated
			// Ensures we scroll after the browser paints new content
			requestAnimationFrame(() => {
				containerRef!.scrollTop = containerRef!.scrollHeight
			})
		}
	}

	onMount(() => {
		scrollToBottom()
	})

	createEffect(() => {
		// @ts-expect-error - Intentionally unused variable
		const _ = props.messages
		scrollToBottom()
	})

	return (
		<section class="flex flex-grow flex-col min-h-0 pl-0 md:pl-8">
			<SectionHeader title="gossip" />
			<div ref={containerRef} class="overflow-scroll overflow-x-hidden flex-grow min-h-0 max-h-[50vh] md:max-h-none" id="gossip-container">
				<ul class="list-none">
					{props.messages?.length > 0 ? (
						props.messages.map((msg) => {
							return (
								<li>
									<NodeMessage id={msg.node} message={msg.message} />
								</li>
							)
						})
					) : (
						<span>&lt; FETCHING GOSSIP &gt;</span>
					)}
				</ul>
			</div>
		</section>
	)
}

const MultiLineChart = (props: { data: LeaderboardResponse }) => {
	let svgRef: SVGSVGElement | undefined

	// Props aren't automatically reactive, so we need to turn the props into a signal.
	// createMemo automatically updates when props.data changes, opposed to createSignal which is manually reactive.
	const chartData = createMemo(() => props.data)

	// TODO: Set dimensions magically
	const margin = { top: 10, right: 10, bottom: 30, left: 50 }
	const width = 400 - margin.left - margin.right
	const height = 300 - margin.top - margin.bottom

	onMount(() => {
		drawChart()
	})

	createEffect(() => {
		updateChart()
	})

	function drawChart() {
		if (!svgRef) {
			return
		}

		const svg = d3
			.select(svgRef)
			.attr("width", width + margin.left + margin.right)
			.attr("height", height + margin.top + margin.bottom)
			.append("g")
			.attr("transform", `translate(${margin.left},${margin.top})`) // push the chart a bit to account for labels/legends

		// Define scales.
		// Scales map data space into SVG space.
		const xScale = d3.scaleLinear().domain([0, 0]).range([0, width])
		const yScale = d3.scaleLinear().domain([0, 0]).range([height, 0])

		// Set scales based on data.
		const allXs = chartData().leaders.flatMap((leader) => leader.values.map((d) => d.x))
		const allYs = chartData().leaders.flatMap((leader) => leader.values.map((d) => d.y))
		yScale.domain([d3.min(allYs)!, d3.max(allYs)!])
		xScale.domain([d3.min(allXs)!, d3.max(allXs)!])

		const xTicksCount = Math.min(5, allXs.length)

		// Horizontal grid lines
		// Setting a tick size to a negative width here extends the ticks across the chart.
		svg.append("g")
			.attr("class", "grid")
			.call(
				d3
					.axisLeft(yScale)
					.tickSize(-width)
					.tickFormat(() => "")
					.ticks(5),
			)
			.selectAll(".tick line")
			.attr("stroke", () => "rgba(0, 0, 0, 0.3)")
			.attr("stroke-dasharray", "3,3")
			.filter((_, i, _nodes) => i === 6)
			.attr("stroke", "none")

		// Vertical grid lines
		svg.append("g")
			.attr("class", "grid")
			.attr("transform", `translate(0,${height})`)
			.call(
				d3
					.axisBottom(xScale)
					.tickSize(-height)
					.tickFormat(() => "")
					.ticks(xTicksCount),
			)
			.selectAll("line")
			.attr("stroke", () => "rgba(0, 0, 0, 0.3)")
			.attr("stroke-dasharray", "3,3")

		// Define color scale
		const color = (i: number) => {
			const cs = ["#2A0D04", "#14BD4A", "#B800C4", "#FF6600", "#00A8F3", "#FFD700", "#8B4513", "#FF1493", "#7FFF00", "#9400D3"]
			return cs[i % cs.length]
		}

		// Define line generator.
		// Given data points, it generates an SVG path.
		const line = d3
			.line<{ x: number; y: number }>()
			.x((d) => xScale(d.x))
			.y((d) => yScale(d.y))

		// Draw data.
		svg.selectAll(".line")
			.data(chartData().leaders)
			.enter()
			.append("path")
			.attr("class", "line")
			.attr("fill", "none")
			.attr("stroke", (_, i) => color(i))
			.attr("stroke-width", 2)
			.attr("stroke-dasharray", "2 2")
			.attr("stroke-linejoin", "round")
			.attr("d", (d) => line(d.values) as string)

		// Add X and Y axis
		svg.append("g")
			.attr("class", "x-axis")
			.attr("transform", `translate(0,${height})`)
			.call(d3.axisBottom(xScale).tickSize(0).tickPadding(10))
			.call(
				d3
					.axisBottom(xScale)
					.tickSize(0)
					.tickFormat((d) => `${Math.round(d.valueOf())}`)
					.ticks(xTicksCount),
			)
			.attr("stroke-width", 2)
			.selectAll("text")
			.style("font-size", "12px")
			.style("font-family", "monospace")

		svg.append("g").attr("class", "y-axis").attr("stroke-width", 2).call(d3.axisLeft(yScale).tickSize(0).tickPadding(10)).selectAll("text").style("font-size", "12px").style("font-family", "monospace")

		// Removes the top-most and right-most borders of the chart
		svg.selectAll(".domain")
			.filter((_, i, _nodes) => i == 0 || i == 1)
			.attr("stroke", "none")

		// Add X-axis label
		svg.append("text")
			.attr("class", "x-axis-label")
			.attr("x", width / 2) // Center it horizontally
			.attr("y", height + margin.bottom) // Position it below the X-axis
			.attr("text-anchor", "middle") // Center the text
			.style("font-size", "10px")
			.style("font-family", "Aux Mono")
			.text("SECONDS ELAPSED")
	}

	function updateChart() {
		if (!svgRef) {
			return
		}
		// Remove previous chart before re-adding
		// This makes the chart update less smooth, but removes any issues with re-rendering.
		d3.select(svgRef).selectAll("*").remove()
		drawChart()
	}

	return <svg ref={svgRef}></svg>
}

function Leaderboard(props: { leaders: { id: string; score: number; index?: number }[] }) {
	return (
		<>
			<ul class="list-none uppercase">
				{props.leaders.map((leader, i) => {
					return (
						<li data-testid={`leader-${i}`}>
							<span class="text-gensyn-green">[{leader.index || i}]</span>&nbsp;
							<span>
								{leader.id} [{leader.score}]
							</span>
						</li>
					)
				})}
			</ul>
		</>
	)
}

function LeaderboardSection(props: { leaders: LeaderboardResponse | undefined; currentRound: number; currentStage: number }) {
	const ctx = useSwarm()
	const [searchQuery, setSearchQuery] = createSignal("")
	const [activeSearchId, setActiveSearchId] = createSignal("")

	const searchedParticipant = createMemo(() => {
		const id = activeSearchId()
		if (!id) {
			return null
		}

		const participants = ctx.participantsById()
		return participants ? participants[id] : null
	})

	const leaderboardSubtitle = createMemo(() => {
		let out = ""
		if (props.currentRound >= 0) {
			out = out + `: Round ${props.currentRound}`

			if (props.currentStage >= 0) {
				out = out + `, stage ${props.currentStage}`
			}
		}

		return out
	})

	const LeaderboardTooltipMessage = () => (
		<>
			<p class="uppercase">Models in the swarm receive rewards based on the following criteria:</p>
			<ul class="mt-4 uppercase">
				<li class="mb-2">
					<strong>Formatted &rarr;</strong> does the model generate output matching the specified format?
				</li>
				<li class="mb-2">
					<strong>Correct &rarr;</strong> is the final answer mathematically correct and formatted correctly?
				</li>
				<li>
					<strong>Insightful &rarr;</strong> in stages requiring reference to best messages from prior rounds, does the model reference those messages, and do they meet the reward criteria for that round?
				</li>
			</ul>
			<p class="uppercase text-center mt-4 mb-4">* * *</p>
			<p class="uppercase">This graph displays the cumulative reward for each node from the moment the page is loaded, not the full history from the start of a round.</p>
		</>
	)

	return (
		<section class="flex flex-grow flex-col min-h-0 border-dashed border-gensyn-brown border-b md:border-b-0 pb-4 mb:pb-0 md:border-r md:pr-8">
			<div class="flex-none">
				<SectionHeader title="cumulative reward" tooltip={<LeaderboardTooltipMessage />} />
				{props.leaders ? <MultiLineChart data={props.leaders} /> : <span class="block w-[400px]">&lt; FETCHING LEADERS &gt;</span>}
			</div>

			<div class="flex-none mt-8">
				<SectionHeader title={`leaderboard ${leaderboardSubtitle()}`} />
			</div>

			<div id="leaderboard-container" class="mt-0 overflow-auto overflow-x-hiddden flex-grow min-h-0">
				<Leaderboard leaders={props.leaders?.leaders || []} />
				<form
					onSubmit={(e) => {
						e.preventDefault()
						setActiveSearchId(searchQuery())
					}}
					class="flex uppercase mt-2 mb-2"
				>
					<input placeholder="ENTER YOUR NODE ADDRESS" type="text" value={searchQuery()} onInput={(e) => setSearchQuery(e.currentTarget.value)} class="border border-gensyn-brown p-2 flex-grow focus:outline-none focus:ring-0 focus:border-gensyn-green" />
					<button type="submit" class="border-t border-b border-r border-gensyn-brown p-2 bg-[rgba(0,0,0,0.05)]">
						Search
					</button>
				</form>
				<div data-testid="leaderboard-search-results">
					<Show when={searchedParticipant()}>
						<Leaderboard leaders={[searchedParticipant()!]} />
					</Show>
				</div>
			</div>
		</section>
	)
}
