import discord
import os
import albhed

client = discord.Client()

@client.event
async def on_message(message):
	if message.author == client.user:
		return

	if message.content == 'ELL':
		await message.add_reaction('📠')
	elif message.content.startswith('!ab '):
		text = message.content[4:]
		await message.channel.send(albhed.translate_to_al_bhed(text))

client.run(os.environ['DISCORD_TOKEN'])